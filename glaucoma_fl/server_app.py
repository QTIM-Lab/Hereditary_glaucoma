"""pytorchexample: A Flower / PyTorch app."""

from typing import List, Tuple, Optional
import os
from collections import OrderedDict

import numpy as np
import torch

from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays, Scalar, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from glaucoma_fl.task import Net, get_weights


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aucs = [num_examples * m["auc_score"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples), "auc_score": sum(aucs) / sum(examples),
            'num_examples': examples, 'auc_not_aggregated': [m["auc_score"] for _, m in metrics]}


class SaveModelStrategy(FedAvg):

    def __init__(self, out_folder, **kwargs):
        super().__init__(**kwargs)
        self.net = Net()
        self.out_folder = out_folder
        os.makedirs(self.out_folder, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: list,
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            params_dict = zip(self.net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

            # Save the model to disk
            torch.save(self.net.state_dict(), f"{self.out_folder}/model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    out_folder = context.run_config["out-folder"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = SaveModelStrategy(
        out_folder,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2, # require all three devices (clients) to be available - can change it
        min_fit_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
