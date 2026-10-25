import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test, class_weights=None):

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    model.eval()

    with torch.no_grad():

        outputs = model(X_test)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # Calculate loss
        if class_weights is not None:

            weights = torch.tensor(
                class_weights,
                dtype=torch.float32
            )

            criterion = nn.CrossEntropyLoss(
                weight=weights
            )

        else:
            criterion = nn.CrossEntropyLoss()

        loss = criterion(
            outputs,
            y_test_tensor
        ).item()

    predictions = predictions.numpy()
    probabilities = probabilities.numpy()

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }