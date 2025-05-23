import matplotlib.pyplot as plt
import pandas as pd


def plot_forecasts(Y_train_df, Y_test_df, forecasts, id, sku, site, filename=None):
    # Plot predictions
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    Y_hat_df = forecasts.reset_index(drop=False).drop(columns=["unique_id", "ds"])
    plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
    plot_df = pd.concat([Y_train_df, plot_df])

    plot_df = plot_df[plot_df.unique_id == id].drop("unique_id", axis=1)
    plt.plot(plot_df["ds"], plot_df["y"], c="black", label="True")
    plt.plot(plot_df["ds"], plot_df["TSMixer-median"], c="blue", label="median")
    plt.fill_between(
        x=plot_df["ds"][-52:],
        y1=plot_df["TSMixer-lo-90"][-52:].values,
        y2=plot_df["TSMixer-hi-90"][-52:].values,
        alpha=0.4,
        label="level 90",
    )
    ax.set_title(f"CAFR Demand Forecast SKU = {sku}, {site}", fontsize=22)
    ax.set_ylabel("Weekly Sales", fontsize=20)
    ax.set_xlabel("Year", fontsize=20)
    ax.legend(prop={"size": 15})
    ax.grid()
    if filename:
        fig.savefig(filename, bbox_inches='tight')