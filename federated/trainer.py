import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def train_model(
    model,
    X_train,
    y_train,
    class_weights,
    epochs=30,
    learning_rate=0.001,
    batch_size=64,
    seed=None
):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)

    # --------------------------------------------------
    # Deterministic DataLoader
    # --------------------------------------------------
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in dataloader:

            outputs = model(X_batch)

            loss = criterion(
                outputs,
                y_batch
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_size_actual = X_batch.size(0)

            total_loss += (
                loss.item() * batch_size_actual
            )

            total_samples += batch_size_actual

        average_loss = (
            total_loss / total_samples
        )

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {average_loss:.4f}"
        )

    return model, average_loss