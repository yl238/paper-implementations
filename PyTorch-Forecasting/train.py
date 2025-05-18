from datetime import date

import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .dataset import create_sequential_dataset, generate_train_and_test
from .LSTM import LSTMModel


def get_apple_stock_data():
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = "2008-01-01"

    df = yf.download("AAPL", start_date, end=end_date)
    df.columns = ["Close", "High", "Low", "Open", "Volume"]
    return df["Open"]


def train_and_evaluate(
    model: nn.Module,
    optimizer: optim,
    loss_fn,
    train_loader,
    test_loader,
    num_epochs,
):
    train_hist = []
    test_hist = []

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        model.train()
        for batch_X, batch_y in train_loader:
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_hist.append(average_loss)

        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0

            for batch_X_test, batch_y_test in test_loader:
                predictions_test = model(batch_X_test)
                test_loss = loss_fn(predictions_test, batch_y_test)
                total_test_loss += test_loss.item()

            average_test_loss = total_test_loss / len(test_loader)
            test_hist.append(average_test_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] - Training loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}"
            )
    return model


if __name__ == "__main__":
    data = get_apple_stock_data()
    dataset_train, dataset_test = generate_train_and_test(
        data,
        train_test_split=0.8,
    )
    # Min-max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(dataset_train)
    scaled_test = scaler.transform(dataset_test)

    X_train, y_train = create_sequential_dataset(scaled_train)
    X_test, y_test = create_sequential_dataset(scaled_test)

    # Set LSTM parameters
    input_size = 1
    num_layers = 3
    hidden_size = 128
    output_size = 1
    dropout = 0.2

    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Learning rate

    batch_size = 32

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trained_model = train_and_evaluate(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=100,
    )
