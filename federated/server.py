import os

import flwr as fl
import numpy as np
import torch

from federated.model import StrokeMLP

from federated.dataset import load_hospital_partitions
from federated.client import StrokeClient
from federated.seed import set_seed
from federated.hash_model import hash_model_parameters
from blockchain.logger import log_round_to_chain
from db.db_utils import insert_hospital_metrics
from db.db_utils import (
    insert_fl_round,
    insert_hospital_metrics
)


NUM_HOSPITALS = 4
NUM_ROUNDS = int(os.environ.get("HEALTHFL_NUM_ROUNDS", "10"))
LOCAL_EPOCHS = int(os.environ.get("HEALTHFL_LOCAL_EPOCHS", "1"))
SEED = 42

SAVE_MODEL_ONLY = os.environ.get(
    "HEALTHFL_SAVE_MODEL_ONLY",
    "1"
) == "0"


set_seed(SEED)


# ---------------------------------------------------------
# Load hospital datasets
# ---------------------------------------------------------

partitions = load_hospital_partitions(
    num_hospitals=NUM_HOSPITALS
)


# ---------------------------------------------------------
# Create Flower clients
# ---------------------------------------------------------

def client_fn(context):

    client_id = int(
        context.node_config["partition-id"]
    )

    hospital = partitions[client_id]

    print(
        f"\nStarting Hospital {client_id}"
    )

    client = StrokeClient(
        X_train=hospital["X_train"],
        y_train=hospital["y_train"],
        X_test=hospital["X_test"],
        y_test=hospital["y_test"],
        class_weights=hospital["class_weights"],
        local_epochs=LOCAL_EPOCHS,
        seed=SEED,
        client_id=client_id
    )

    return client.to_client()


# ---------------------------------------------------------
# Aggregate evaluation metrics
# ---------------------------------------------------------

def weighted_average(metrics):

    total_samples = sum(
        num_samples
        for num_samples, _ in metrics
    )

    if total_samples == 0:
        return {}

    aggregated = {}

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]

    for metric_name in metric_names:

        weighted_sum = sum(
            num_samples * metric_dict[metric_name]
            for num_samples, metric_dict in metrics
        )

        aggregated[metric_name] = (
            weighted_sum / total_samples
        )

    return aggregated


# ---------------------------------------------------------
# Send round number to clients
# ---------------------------------------------------------

def fit_config(server_round):

    return {
        "server_round": server_round
    }


# ---------------------------------------------------------
# Save global model
# ---------------------------------------------------------

def save_global_model(parameters, feature_names, round_number):

    model = StrokeMLP(
        input_size=len(feature_names)
    )

    state_dict = model.state_dict()

    for key, value in zip(
        state_dict.keys(),
        parameters
    ):
        state_dict[key] = torch.tensor(
            value,
            dtype=state_dict[key].dtype
        )

    model.load_state_dict(state_dict)

    save_path = (
        "saved_models/"
        "final_global_model.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": feature_names,
            "round": round_number
        },
        save_path
    )

    print(
        f"Global model saved to: {save_path}"
    )


# ---------------------------------------------------------
# Blockchain-enabled FedAvg strategy
# ---------------------------------------------------------

class BlockchainFedAvg(
    fl.server.strategy.FedAvg
):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        # Perform normal FedAvg aggregation
        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures
            )
        )

        if aggregated_parameters is not None:

            # Convert aggregated parameters to NumPy arrays
            model_parameters = (
                fl.common.parameters_to_ndarrays(
                    aggregated_parameters
                )
            )

            # Generate SHA-256 hash
            model_hash = hash_model_parameters(
                model_parameters
            )

            print(
                f"\nRound {server_round} "
                f"Global Model Hash:"
            )

            print(model_hash)

            # Save final global model
            if server_round == NUM_ROUNDS:

                save_global_model(
                    parameters=model_parameters,
                    feature_names=partitions[0]["feature_names"],
                    round_number=server_round
                )

            print(
                f"\nRound {server_round} "
                f"Global Model Hash:"
            )

            print(model_hash)

            if not SAVE_MODEL_ONLY:

                # Store hash on blockchain
                blockchain_result = log_round_to_chain(
                    round_number=server_round,
                    model_hash=model_hash
                )

                transaction_hash = blockchain_result[
                    "transaction_hash"
                ]

                block_number = blockchain_result[
                    "block_number"
                ]

                print(
                    f"Blockchain Transaction: "
                    f"{transaction_hash}"
                )

                # -------------------------------------------------
                # Store round information in SQLite
                # -------------------------------------------------

                insert_fl_round(
                    round_id=server_round,
                    num_hospitals=NUM_HOSPITALS,
                    global_model_hash=model_hash,
                    blockchain_tx_hash=transaction_hash,
                    blockchain_block_number=block_number,
                    status="completed"
                )

                print(
                    f"Round {server_round} "
                    f"stored in SQLite."
                )

        return (
            aggregated_parameters,
            aggregated_metrics
        )

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures
    ):

        print(
            f"\n--- Hospital Metrics "
            f"for Round {server_round} ---"
        )

        for client_proxy, evaluate_result in results:

            hospital_metrics = evaluate_result.metrics

            hospital_id = int(
                hospital_metrics["hospital_id"]
            )

            loss = float(
                evaluate_result.loss
            )

            accuracy = float(
                hospital_metrics["accuracy"]
            )

            precision = float(
                hospital_metrics["precision"]
            )

            recall = float(
                hospital_metrics["recall"]
            )

            f1 = float(
                hospital_metrics["f1"]
            )

            auc = float(
                hospital_metrics["auc"]
            )

            # Print metrics
            print(
                f"Hospital {hospital_id}: "
                f"Loss={loss:.4f}, "
                f"Accuracy={accuracy:.4f}, "
                f"Precision={precision:.4f}, "
                f"Recall={recall:.4f}, "
                f"F1={f1:.4f}, "
                f"AUC={auc:.4f}"
            )

            # Get hospital partition
            hospital = partitions[hospital_id]

            # Store hospital metrics in SQLite
            insert_hospital_metrics(
                round_id=server_round,
                hospital_id=hospital_id,
                num_train_samples=len(
                    hospital["X_train"]
                ),
                num_test_samples=len(
                    hospital["X_test"]
                ),
                loss=loss,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                auc=auc
            )

        # Perform normal FedAvg metric aggregation
        aggregated_loss, aggregated_metrics = (
            super().aggregate_evaluate(
                server_round,
                results,
                failures
            )
        )

        return (
            aggregated_loss,
            aggregated_metrics
        )


# ---------------------------------------------------------
# Create strategy
# ---------------------------------------------------------

strategy = BlockchainFedAvg(

    fraction_fit=1.0,

    fraction_evaluate=1.0,

    min_fit_clients=NUM_HOSPITALS,

    min_evaluate_clients=NUM_HOSPITALS,

    min_available_clients=NUM_HOSPITALS,

    evaluate_metrics_aggregation_fn=weighted_average,

    on_fit_config_fn=fit_config
)


# ---------------------------------------------------------
# Start Flower simulation
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print(
        "HEALTHFL-CHAIN FEDERATED LEARNING"
    )
    print("=" * 60)

    print(
        f"Hospitals      : {NUM_HOSPITALS}"
    )

    print(
        f"FL rounds      : {NUM_ROUNDS}"
    )

    print(
        f"Local epochs   : {LOCAL_EPOCHS}"
    )

    print(
        f"Random seed    : {SEED}"
    )

    print(
        "Strategy       : BlockchainFedAvg"
    )

    print(
        "\nStarting Flower simulation...\n"
    )

    fl.simulation.start_simulation(

        client_fn=client_fn,

        num_clients=NUM_HOSPITALS,

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy
    )