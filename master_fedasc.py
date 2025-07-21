from .master import Master
import torch
import torch.distributed as dist
from pcode.utils.tensor_buffer import TensorBuffer
import numpy as np


class MasterFedASC(Master):
	def __init__(self, conf):
		super().__init__(conf)
		self.deformation = {client_id: 0.0 for client_id in self.client_ids}
		self.deformation_hyperparameters = conf.deformation_hyperparameters

	def run(self):
		for comm_round in range(1, 1 + self.conf.n_comm_rounds):
			self.conf.graph.comm_round = comm_round
			self.conf.logger.log(
				f"Master starting one round of federated learning: (comm_round={comm_round})."
			)

			# Log current deformation values before the round starts
			self.conf.logger.log(f"Current deformation before round {comm_round}: {self.deformation}")

			# get random n_local_epochs. List of local training epochs
			list_of_local_n_epochs = get_n_local_epoch(
				conf=self.conf, n_participated=self.conf.n_participated
			)
			self.list_of_local_n_epochs = list_of_local_n_epochs

			# random select clients from a pool.
			selected_client_ids = self._random_select_clients()

			# detect early stopping.
			self._check_early_stopping()

			# init the activation tensor and broadcast to all clients (either start or stop).
			self._activate_selected_clients(
				selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs
			)

			# will decide to send the model or stop the training.
			if not self.conf.is_finished:
				# broadcast the models and parameters to activated clients.
				self._send_model_and_weights_to_selected_clients(selected_client_ids)
			else:
				dist.barrier()
				self.conf.logger.log(
					f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
				)
				return

			# wait to receive the local models and parameters.
			flatten_local_models, activated_deformation = self._receive_models_weights_from_selected_clients(
				selected_client_ids
			)
			# Update deformation
			extended_data_quantity_sum = sum(ext for (con, ext) in activated_deformation.values())
			activated_deformation = {
    			client_id: con * ext / extended_data_quantity_sum * self.deformation_hyperparameters
    			for client_id, (con, ext) in activated_deformation.items()
			}

			self.deformation.update(activated_deformation)
			self.conf.logger.log(f"Deformation received from clients: {activated_deformation}")
			self.fedavg_weights = compute_fedavg_weights_from_deformation(activated_deformation)

			# aggregate the local models and evaluate on the validation dataset.
			self._aggregate_model_and_evaluate(flatten_local_models)

			# evaluate the aggregated model.
			self.conf.logger.log(f"Master finished one round of federated learning.\n")

		# formally stop the training (the master has finished all communication rounds).
		dist.barrier()
		self._finishing()

	def _send_model_and_weights_to_selected_clients(self, selected_client_ids):
		self.conf.logger.log("Master sends the models and weights to workers.")
		# Use real weights for decay after each node has been activated once
		use_real_weights = all(value != 0.0 for value in self.deformation.values())

		# Extract subset
		selected_client_deformations = {
			cid: self.deformation[cid] for cid in selected_client_ids
		}
		deformation_values = list(selected_client_deformations.values())
		min_deformation = min(deformation_values)
		max_deformation = max(deformation_values)

		# Linear inverse mapping weights (not normalized)
		if min_deformation == max_deformation:
			normalized_weights = {cid: 1.0 for cid in selected_client_ids}
		else:
			normalized_weights = {
				# cid: (deformation - min_deformation) / (max_deformation - min_deformation)
				cid: deformation / max_deformation
				for cid, deformation in selected_client_deformations.items()
			}
		
		for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
			arch = self.clientid2arch[selected_client_id]
			client_model_state_dict = self.client_models[arch].state_dict()
			flatten_model = TensorBuffer(list(client_model_state_dict.values()))

			if use_real_weights:
				client_weight = torch.tensor(
					[normalized_weights[selected_client_id]], dtype=torch.float32
				)
			else:
				client_weight = torch.tensor([0.0], dtype=torch.float32)

			combined_buffer = torch.cat([flatten_model.buffer, client_weight])

			dist.send(tensor=combined_buffer, dst=worker_rank)
			self.conf.logger.log(
				f"\tMaster sent the current model={arch} and deformation weight={client_weight.item():.4f} to process_id={worker_rank}."
			)

		dist.barrier()

	def _receive_models_weights_from_selected_clients(self, selected_client_ids):
		self.conf.logger.log(f"Master waits to receive the local models.")
		dist.barrier()

		# Initialize placeholders for receiving local models and accuracies.
		flatten_local_models = dict()
		client_deformations = dict()
		reqs = []

		# Prepare buffers for model parameters and top1_accuracy.
		for selected_client_id in selected_client_ids:
			arch = self.clientid2arch[selected_client_id]
			client_tb = TensorBuffer(
				list(self.client_models[arch].state_dict().values())
			)
			# Add space for the additional accuracy value
			combined_buffer_size = client_tb.buffer.numel() + 2
			combined_buffer = torch.zeros(combined_buffer_size, dtype=torch.float32)
			flatten_local_models[selected_client_id] = combined_buffer

		# Asynchronously receive combined buffer (model + accuracy) from clients.
		for client_id, world_id in zip(selected_client_ids, self.world_ids):
			req = dist.irecv(
				tensor=flatten_local_models[client_id], src=world_id
			)
			reqs.append(req)

		for req in reqs:
			req.wait()

		dist.barrier()
		self.conf.logger.log(f"Master received all local models.")

		# Extract models and accuracies from the combined buffer.
		for client_id in selected_client_ids:
			combined_buffer = flatten_local_models[client_id]
			model_buffer = combined_buffer[:-2]
			# Model parameters
			contraction_coefficient = combined_buffer[-2].item()
			extended_data_quantity = combined_buffer[-1].item()

			# Reconstruct the TensorBuffer for the model
			arch = self.clientid2arch[client_id]
			client_tb = TensorBuffer(
				list(self.client_models[arch].state_dict().values())
			)
			client_tb.buffer.copy_(model_buffer)

			# Save the reconstructed model and accuracy
			flatten_local_models[client_id] = client_tb
			client_deformations[client_id] = (contraction_coefficient, extended_data_quantity)

		return flatten_local_models, client_deformations
		
def get_n_local_epoch(conf, n_participated):
	if conf.min_local_epochs is None:
		return [conf.local_n_epochs] * n_participated
	else:
		# here we only consider to (uniformly) randomly sample the local epochs.
		assert conf.min_local_epochs > 1.0
		random_local_n_epochs = conf.random_state.uniform(
			low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
		)
		return random_local_n_epochs


def compute_fedavg_weights_from_deformation(client_deformations, temperature=1.0):
	if temperature <= 0:
		raise ValueError("Softmax temperature must be positive.")
	
	clients = list(client_deformations.keys())
	deformation_array = np.array([client_deformations[client] for client in clients])

	if np.allclose(deformation_array, deformation_array[0]):
		# All deformations are equal, distribute evenly
		return {client: 1.0 / len(clients) for client in clients}

	# Use softmax(deformation / T)
	softmax_weights = np.exp(deformation_array / temperature)
	normalized_weights = softmax_weights / np.sum(softmax_weights)
	
	return {
		client: weight for client, weight in zip(clients, normalized_weights)
	}