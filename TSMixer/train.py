import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import TSMixer
from plotting import plot_forecasts


def generate_training_data(
    input_file, category, start_date, end_date, n_samples=1000
):
    df = pd.read_parquet(input_file)

    df["unique_id"] = df["SourceID"] + "_" + df["SiteCode"]
    unique_product_sites = df[
        ["unique_id", "SourceID", "SiteName"]
    ].drop_duplicates()

    df["WeekStart"] = pd.to_datetime(df["WeekStart"])
    df_val = df[(df["WeekStart"] > start_date) & (df["WeekStart"] <= end_date)]

    df_total_sales = (
        df_val.groupby("unique_id")
        .agg({"Quantity": "sum"})
        .sort_values(by="Quantity", ascending=False)
    )
    sampled_ids = (
        df_total_sales[df_total_sales["Quantity"] > 0].sample(n_samples).index.values
    )
    df_val[df_val.unique_id.isin(tuple(sampled_ids))][
        ["SourceID", "SiteCode", "SiteName"]
    ].drop_duplicates().to_csv(f"data/{category}_sample_ids.csv", index=False)

    df_section = df_val[["WeekStart", "Quantity", "unique_id"]]
    df_section = df_section[df_section["unique_id"].isin(tuple(sampled_ids))]

    df_section.reset_index().drop(columns=["index"], inplace=True)
    df_section.columns = ["ds", "y", "unique_id"]

    df_section.groupby("unique_id").agg({"y": "sum"}).sort_values(
        by="y", ascending=False
    ).to_csv(f"data/{category}_total_product_sales.csv")

    return df_section


def train_model(
    df, n_series, input_size=52, forecast_horizon=52, n_test=52, model_prefix="test"
):
    Y_train_df = df[df.ds < df["ds"].values[-n_test]].reset_index(drop=True)
    Y_test_df = df[df.ds >= df["ds"].values[-n_test]].reset_index(drop=True)

    model = TSMixer(
        h=forecast_horizon,
        input_size=input_size,
        n_series=n_series,
        n_block=4,
        ff_dim=4,
        dropout=0.1,
        revin=True,
        scaler_type="standard",
        max_steps=500,
        early_stop_patience_steps=5,
        val_check_steps=5,
        learning_rate=1e-3,
        loss=MQLoss(),
        batch_size=32,
    )
    fcst = NeuralForecast(models=[model], freq="7D")
    fcst.fit(df=Y_train_df, val_size=n_test)

    fcst.save(
        path=f"./checkpoints/{model_prefix}_run/",
        model_index=None,
        overwrite=True,
        save_dataset=True,
    )
    return fcst, Y_train_df, Y_test_df


def threshold_zero(forecasts):
    forecasts["TSMixer-median"][forecasts["TSMixer-median"] < 0] = 0
    forecasts["TSMixer-hi-90"][forecasts["TSMixer-hi-90"] < 0] = 0
    forecasts["TSMixer-lo-90"][forecasts["TSMixer-lo-90"] < 0] = 0
    return forecasts


if __name__ == "__main__":
    file = "data/CAFR_sales_weekly_sales_sales_13_LIGHTING.parquet"

    category = "13_LIGHTING"
    start_date = "2021-06-01"
    end_date = "2025-01-01"

    n_samples = 20000
    h = 52

    df = generate_training_data(
        input_file=file,
        category=category,
        start_date=start_date,
        end_date=end_date,
        n_samples=n_samples,
    )

    model, Y_train_df, Y_test_df = train_model(
        df, n_series=n_samples, forecast_horizon=h, n_test=h, model_prefix=category
    )

    forecasts = model.predict(futr_df=Y_test_df)
    forecasts.to_parquet(f"data/{category}_forecasts.parquet")

    forecasts = threshold_zero(forecasts)

    id = Y_test_df["unique_id"].values[0]
    source = id.split("_")[0]
    site = id.split("_")[1]

    plot_forecasts(
        Y_train_df,
        Y_test_df,
        forecasts,
        id=id,
        sku=source,
        site=site,
        filename=f"figures/{category}_forecasts_{id}.png",
    )
