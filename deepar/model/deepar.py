import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Multi-Layer Perceptron Decoder

    Args:
       in_features (_type_): _description_
       output_features (_type_): _description_
       hidden_size (_type_): _description_
       hidden_layers (_type_): _description_

    """

    def __init__(self, in_features, output_features, hidden_size, hidden_layers):
        super().__init__()

        if hidden_layers == 0:
            # Input layer
            layers = [
                nn.Linear(in_features=in_features, out_features=output_features)
            ]
        else:
            # Input layer
            layers = [
                nn.Linear(in_features=in_features, out_features=hidden_size),
                nn.ReLU(),
            ]
            # Hidden layers
            for i in range(hidden_layers - 2):
                layers += [
                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    nn.ReLU(),
                ]
            # Output layer
            layers += [
                nn.Linear(in_features=hidden_size, out_features=output_features)
            ]

        # Store in layers as ModuleList
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeepAR(nn.Module):
    def __init__(
        self,
        h,
        input_size: int = -1,
        h_train: int = 1,
        lstm_n_layers: int = 2,
        lstm_hidden_size: int = 128,
        lstm_dropout: float = 0.1,
        decoder_hidden_layers: int = 0,
        decoder_hidden_size: int = 0,
        learning_rate: float = 1.0e-3,
        max_steps: int = 1000,
        batch_size: int = 32,
        step_size: int = 1,
        random_seed: int = 1,
        optimizer=None,
    ):
        # LSTM input size (1 for target variable y)
        input_encoder = 1

        # LSTM
        self.encoder_n_layers = lstm_n_layers
        self.encoder_hidden_size = lstm_hidden_size
        self.encoder_dropout = lstm_dropout

        # Instantiate model
        self.rnn_state = None
        self.maintain_state = False
        self.hist_encoder = nn.LSTM(
            input_size=input_encoder,
            hidden_size=self.encoder_hidden_size,
            num_layers=self.encoder_n_layers,
            dropout=self.encoder_dropout,
            batch_first=True,
        )

        # Decoder MLP
        self.decoder = Decoder(
            in_features=lstm_hidden_size,
            output_features=self.loss.outputsize_multiplier,
            hidden_size=decoder_hidden_size,
            hidden_layers=decoder_hidden_layers
        )
   