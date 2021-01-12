from torch import nn


class Model(nn.Module):
    def __init__(self, input_size=4, hidden_layers=2, hidden_layer_size=8, output_size=2):
        super(Model, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_layer_size, hidden_layer_size)
             for i in range(hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = nn.LeakyReLU()(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = nn.LeakyReLU()(x)

        x = self.output_layer(x)

        return x
