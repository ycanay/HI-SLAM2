import torch


class Predictor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Predictor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.output(x)
        return x
