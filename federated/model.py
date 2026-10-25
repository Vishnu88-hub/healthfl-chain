import torch
import torch.nn as nn


class StrokeMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)




if __name__ == "__main__":

    input_size = 10

    model = StrokeMLP(input_size)

    print(model)

    sample_input = torch.randn(1, input_size)

    output = model(sample_input)

    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
    print("Output:", output)