from black import out
import torch


class RecurrentNetwork(torch.nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, n_layers: int
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.layers.append(torch.nn.GRU(input_size, hidden_size))
            else:
                self.layers.append(torch.nn.GRU(hidden_size, hidden_size))
        self.output_layer = torch.nn.Linear(
            in_features=hidden_size, out_features=output_size
        )

    def __call__(self, x):

        y = x.transpose(dim0=1, dim1=0)
        for i, layer in enumerate(self.layers):
            if i < self.n_layers - 1:
                y, _ = layer(y)
            else:
                _, y = layer(y)
        output = self.output_layer(y).transpose(dim0=1, dim1=0)

        if self.output_size == 1:
            return output
        else:
            return output.squeeze()
