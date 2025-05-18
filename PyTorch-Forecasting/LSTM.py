import torch.nn as nn

class LSTMModel(nn.Module):
    """A simple LSTM-based regression model for time series forecasting.

    This model consists of an LSTM layer followed by a linear layer. 
    It takes a sequence of time steps as input and predicts the next value 
    in the sequence.

    Args:
        input_size (int): Number of features in the input at each time step.
        hidden_size (int): Number of features in the hidden state of the LSTM.
        num_layers (int): Number of recurrent layers in the LSTM.
        dropout (float, optional): Dropout probability for the LSTM layers 
            (default: 0.2).

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

    Forward Output:
        out (torch.Tensor): Output tensor of shape (batch_size, 1),
            the predicted next value for each input sequence.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out