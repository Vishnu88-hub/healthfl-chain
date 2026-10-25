import random
import numpy as np
import torch
import flwr as fl

from federated.model import StrokeMLP
from federated.trainer import train_model
from federated.evaluate import evaluate_model


class StrokeClient(fl.client.NumPyClient):

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        class_weights,
        local_epochs=1,
        seed=42,
        client_id=0
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_weights = class_weights
        self.local_epochs = local_epochs

        self.seed = seed
        self.client_id = client_id

        # --------------------------------------------------
        # Make model initialization identical for all clients
        # --------------------------------------------------
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)

        self.model = StrokeMLP(input_size=X_train.shape[1])

    def get_parameters(self, config):
        return [
            val.detach().cpu().numpy()
            for val in self.model.state_dict().values()
        ]

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        # --------------------------------------------------
        # Deterministic seed for this client + FL round
        # --------------------------------------------------
        server_round = int(config.get("server_round", 0))

        round_seed = (
            self.seed
            + (self.client_id * 1000)
            + server_round
        )

        random.seed(round_seed)
        np.random.seed(round_seed)
        torch.manual_seed(round_seed)

        self.model, loss = train_model(
            model=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            class_weights=self.class_weights,
            epochs=self.local_epochs,
            seed=round_seed
        )

        return (
            self.get_parameters(config),
            len(self.X_train),
            {
                "loss": float(loss)
            }
        )

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        metrics = evaluate_model(
            model=self.model,
            X_test=self.X_test,
            y_test=self.y_test,
            class_weights=self.class_weights
        )

        return (
            float(metrics["loss"]),
            len(self.X_test),
            {
                "hospital_id": self.client_id,
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "auc": float(metrics["auc"])
            }
        )

    def set_parameters(self, parameters):

        state_dict = self.model.state_dict()

        for key, value in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(
                value,
                dtype=state_dict[key].dtype
            )

        self.model.load_state_dict(state_dict)