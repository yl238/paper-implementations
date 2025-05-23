import pandas as pd
import numpy as np

def compute_metrics(dataframe: pd.DataFrame) -> dict:
        """Computes bias, accuracy, and error metrics."""
        dataframe = dataframe.copy()  # Prevent modifying the original dataframe

        # Absolute error and percentage error calculations
        dataframe["AbsoluteError"] = np.abs(dataframe["Quantity"] - dataframe["Predict"])
        dataframe["PercentageError"] = np.where(
            dataframe["Quantity"] == 0, np.nan, np.minimum(1, dataframe["AbsoluteError"] / dataframe["Quantity"])
        )

        # dataframe["SymmetricPercentageError"] = np.abs(dataframe["Quantity"] - dataframe["Predict"]) / (
        #     (np.abs(dataframe["Quantity"]) + np.abs(dataframe["Predict"])) / 2
        # )

        denom = (np.abs(dataframe["Quantity"]) + np.abs(dataframe["Predict"])) / 2
        dataframe["SymmetricPercentageError"] = np.where(
            denom == 0,
            0,  # or np.nan depending on your business rule
            np.abs(dataframe["Quantity"] - dataframe["Predict"]) / denom,
        )

        # Weighted MAPE (wMAPE)
        dataframe["WeightedError"] = dataframe["AbsoluteError"] * dataframe["RetailPrice"]
        dataframe["WeightedQuantity"] = dataframe["Quantity"] * dataframe["RetailPrice"]

        # Aggregate statistics
        total_actual = dataframe["Quantity"].sum()
        total_predicted = dataframe["Predict"].sum()
        total_weighted_actual = dataframe["WeightedQuantity"].sum()
        total_weighted_error = dataframe["WeightedError"].sum()

        overall_bias = (total_predicted - total_actual) / total_actual if total_actual != 0 else np.nan
        overall_1_mape = (1 - dataframe["PercentageError"]).mean()
        overall_2_smape = 2 - dataframe["SymmetricPercentageError"].mean()
        overall_weighted_mape = (
            1 - (total_weighted_error / total_weighted_actual) if total_weighted_actual != 0 else np.nan
        )
        rmse = np.sqrt(np.mean((dataframe["Quantity"] - dataframe["Predict"]) ** 2))
        mae = np.mean(dataframe["AbsoluteError"])

        # Format metrics to 3 decimal places and return as a sorted dictionary
        aggregate_stats = {
            "Overall Bias": round(overall_bias, 3),
            "Overall 1-MAPE": round(overall_1_mape, 3),
            "Overall 2-SMAPE": round(overall_2_smape, 3),
            "Overall 1-Weighted MAPE": round(overall_weighted_mape, 3),
            "Total Actual Sales": round(total_actual, 3),
            "Total Predicted Sales": round(total_predicted, 3),
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
        }

        return dict(sorted(aggregate_stats.items()))  # Sorting metrics alphabetically

# Compute metrics for all data and non-zero actual sales
metrics_all = compute_metrics(data)
metrics_non_zero = compute_metrics(data[data["Quantity"] > 0])

