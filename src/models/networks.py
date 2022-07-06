import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear


class TransformerEncoderModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 2,
        dim_feedforward: int = 512,
        n_layers: int = 2,
    ):
        super().__init__()
        self.input_layer = Linear(in_features=input_dim, out_features=d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.encoder = TransformerEncoder(encoder_layer, n_layers)
        self.output_layer = Linear(in_features=d_model, out_features=output_dim)

    def __call__(self, x):
        y = self.input_layer(x)
        y = self.encoder(y)
        y = y.mean(dim=1)
        return self.output_layer(y)


class MLP(torch.nn.Module):
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
                self.layers.append(torch.nn.Linear(input_size, hidden_size))
            else:
                self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(torch.nn.ReLU())
        self.output_layer = torch.nn.Linear(
            in_features=hidden_size, out_features=output_size
        )

    def __call__(self, x):
        y = torch.nn.Flatten(start_dim=1, end_dim=-1)(x)
        for i, layer in enumerate(self.layers):
            y = layer(y)
        output = self.output_layer(y)
        if self.output_size == 1:
            return output
        else:
            return output.squeeze()


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
