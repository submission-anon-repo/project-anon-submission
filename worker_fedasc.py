from .worker import Worker
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import math
import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.logging import display_training_stat
from pcode.utils.stat_tracker import RuntimeTracker


class WorkerFedASC(Worker):
	def __init__(self, conf):
		super().__init__(conf)
		self.real_class_counts = torch.zeros(self.num_class).to(self.device)
		self.last_batch_expanded = torch.tensor(0.0, device=self.device)
		self.contraction_coefficient_per_epoch = {} 
		self.extended_data_quantity_per_epoch = {}
		self.max_gamma_threshold = getattr(conf, "max_gamma_threshold", 100.0)
	
	def run(self):
		while True:
			self._listen_to_master()

			# check if we need to terminate the training or not.
			if self._terminate_by_early_stopping():
				return
			
			self._recv_model_and_weight_from_master()
			self._train()
			self._send_model_and_weights_to_master()

			# check if we need to terminate the training or not.
			if self._terminate_by_complete_training():
				return
	
	def _train(self):
		self.model.train()
		# init the model and dataloader.
		if self.conf.graph.on_cuda:
			self.model = self.model.cuda()
		self.train_loader, _ = create_dataset.define_data_loader(
			self.conf,
			dataset=self.dataset["train"],
			# localdata_id start from 0 to the # of clients - 1.
			# client_id starts from 1 to the # of clients.
			localdata_id=self.conf.graph.client_id - 1,
			is_train=True,
			data_partitioner=self.data_partitioner,
		)
		
		self.coef_sum = torch.zeros(self.num_class).to(self.device)
		self.coef_count = 0
		self.criterion = lambda outputs, labels: self.class_balanced_cross_entropy(outputs, labels)
		
		# define optimizer, scheduler and runtime tracker.
		self.optimizer = create_optimizer.define_optimizer(
			self.conf, model=self.model, optimizer_name=self.conf.optimizer
		)
		self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
		self.tracker = RuntimeTracker(conf=self.conf, metrics_to_track=self.metrics.metric_names,
		                              use_class_accuracy=self.use_class_accuracy)
		self.conf.logger.log(
			f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round}, max_gamma_threshold={self.max_gamma_threshold})."
		)
		
		# efficient local training.
		if hasattr(self, "model_compression_fn"):
			self.model_compression_fn.compress_model(
				param_groups=self.optimizer.param_groups
			)
		
		# entering local updates and will finish only after reaching the expected local_n_epochs.
		while True:
			# Reset epoch counters
			self.extended_data_quantity_per_epoch[int(self.scheduler.epoch)] = torch.tensor(0.0, device=self.device)
			self.real_class_counts = torch.zeros(self.num_class, device=self.device)
			
			for _input, _target in self.train_loader:				
				# load data
				with self.timer("load_data", epoch=self.scheduler.epoch_):
					data_batch = create_dataset.load_data_batch(
						self.conf, _input, _target, is_training=True, device=str(self.device)
					)
				
				# inference and get current performance.
				with self.timer("forward_pass", epoch=self.scheduler.epoch_):
					self.optimizer.zero_grad()
					loss, output = self._inference(data_batch)
					
					# in case we need self distillation to penalize the local training
					# (avoid catastrophic forgetting).
					self._local_training_with_self_distillation(
						loss, output, data_batch
					)
				
				with self.timer("backward_pass", epoch=self.scheduler.epoch_):
					loss.backward()
					self._add_grad_from_prox_regularized_loss()
					# self._ema_update_gradients()
					# Calculate contraction_coefficient
					self.contraction_coefficient_per_epoch[int(self.scheduler.epoch)] = self.compute_contraction_coefficient()
					self.optimizer.step()
					self.scheduler.step()
				
				# efficient local training.
				with self.timer("compress_model", epoch=self.scheduler.epoch_):
					if hasattr(self, "model_compression_fn"):
						self.model_compression_fn.compress_model(
							param_groups=self.optimizer.param_groups
						)
				
				if self.scheduler.epoch_ % 1 == 0:  # Check if the current epoch is an integer
					# display the logging info.
					display_training_stat(self.conf, self.scheduler, self.tracker)
					self.log_average_coef()
				
				# display tracking time.
				if (
						self.conf.display_tracked_time
						and self.scheduler.local_index % self.conf.summary_freq == 0
				):
					self.conf.logger.log(self.timer.summary())
				
				# check divergence.
				if self.tracker.stat["loss"].avg > 1e3 or np.isnan(
						self.tracker.stat["loss"].avg
				):
					self.conf.logger.log(
						f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
					)
					self._terminate_comm_round()
					return
				
				# check stopping condition.
				if self._is_finished_one_comm_round():
					self._terminate_comm_round()
					return
			
			# refresh the logging cache at the end of each epoch.
			self.tracker.reset()

			if self.conf.logger.meet_cache_limit():
				self.conf.logger.save_json()
	
	def _inference(self, data_batch):
		"""Inference on the given model and get loss and accuracy."""
		# Forward pass: get the output from the model
		# output = self.model(data_batch["input"])
		if self.conf.data in ['sst', 'ag_news']:
			feature, output = self.model(data_batch["input"])
		else:
			output = self.model(data_batch["input"])
		
		# Standard loss computation
		loss = self.criterion(output, data_batch["target"])
		performance, class_correct, class_total = self.metrics.evaluate(
			loss, output, data_batch["target"], use_class_accuracy=self.tracker.use_class_accuracy
		)

		# Update tracker
		if self.tracker is not None:
			if self.use_class_accuracy:
				self.tracker.update_metrics(
					[loss.item()] + (performance if isinstance(performance, list) else [performance]),
					n_samples=data_batch["target"].size(0),
					class_correct=class_correct,
					class_total=class_total,
				)
			else:
				self.tracker.update_metrics(
					[loss.item()] + (performance if isinstance(performance, list) else [performance]),
					n_samples=data_batch["target"].size(0),
				)

		return loss, output

	def _send_model_and_weights_to_master(self):
		dist.barrier()
		flatten_model = TensorBuffer(list(self.model.state_dict().values()))
		# === 1. Calculate the average of contraction and expansion values ===
		avg_contraction = torch.stack(list(self.contraction_coefficient_per_epoch.values())).mean()
		avg_extended = torch.stack(list(self.extended_data_quantity_per_epoch.values())).mean()

		# === 2. Log information ===
		self.conf.logger.log(
			f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending model ({self.arch}) with avg_contraction={avg_contraction.item():.4f}, avg_extended_data={avg_extended.item():.2f}."
		)

		# === 3. Concatenate tensors ===
		weight_tensor = torch.tensor([avg_contraction.item(), avg_extended.item()], dtype=torch.float32)
		combined_buffer = torch.cat([flatten_model.buffer, weight_tensor])  # Combine model and weight
		dist.send(tensor=combined_buffer, dst=0)
		dist.barrier()

	def compute_contraction_coefficient(self, phi=1e-8):
		C = self.num_class
		# Variance term
		variance = ((self.real_class_counts - self.extended_data_quantity_per_epoch[int(self.scheduler.epoch)] / C) ** 2).mean()
		contraction_coefficient = 1.0 / (torch.sqrt(variance) + phi)
		return contraction_coefficient

	def class_balanced_cross_entropy(self, logits, targets):
		"""
		logits: [B, C], model predictions
		targets: [B], ground truth labels
		self.num_class: total number of classes
		"""
		device = logits.device

		# === 1. Class statistics
		class_counts = torch.bincount(targets, minlength=self.num_class).float().to(device)

		# === 1.5 Accumulate actual counts (for cross-epoch statistics)
		self.real_class_counts += class_counts.detach()

		# === 2. Per-class gamma (with maximum threshold limit)
		gamma_per_class = torch.zeros_like(class_counts)
		nonzero_mask = class_counts > 0
		gamma_per_class[nonzero_mask] = class_counts.max() / (class_counts[nonzero_mask] + 1e-8)
		# Apply maximum threshold limit
		gamma_per_class = torch.clamp(gamma_per_class, max=self.max_gamma_threshold)

		# === 3. Per-class coefficient
		coef = torch.zeros_like(class_counts)
		coef[nonzero_mask] = gamma_per_class[nonzero_mask]
		coef = cbl_weights_decay(coef, self.decay_weight)
		self.coef_sum += coef.detach()
		self.coef_count += 1

		# === 4. Expansion quantity calculation
		expanded_sample_count = (coef * class_counts).sum()
		self.last_batch_expanded = expanded_sample_count.item()
		self.extended_data_quantity_per_epoch[int(self.scheduler.epoch)] += self.last_batch_expanded

		# === 5. Weighted cross entropy
		loss_fn = nn.CrossEntropyLoss(weight=coef, reduction='sum')
		loss = loss_fn(logits, targets)
		loss = loss / coef[targets].sum()

		return loss
	
	def log_average_coef(self):
		if self.coef_count > 0:
			avg_coef = self.coef_sum / self.coef_count
			self.conf.logger.log(
				f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) "
				f"Avg class balanced weights: {avg_coef.tolist()}"
			)
			self.coef_sum.zero_()
			self.coef_count = 0


def cbl_weights_decay(weights, decay_weight):
	"""
	Adjust weights by applying decay and ensuring minimum weight is 1.

	Args:
		weights (torch.Tensor): Original weights, shape (num_classes,).
		decay_weight (float): Decay factor to apply to the weights.

	Returns:
		torch.Tensor: Adjusted weights.
	"""
	if decay_weight == 0.0:
		return weights
	# Apply decay factor to the weights
	weights *= decay_weight
	nonzero_mask = weights > 0.0
	weights[nonzero_mask] = torch.clamp(weights[nonzero_mask], min=1.0)

	return weights