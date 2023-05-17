from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from argparse import ArgumentParser

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
    
if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=400
    )
    parser.add_argument(
        "--port",
        type=int,
        defualt=8080
    )
    
    args = parser.parse_args()
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

    # Start Flower server
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )