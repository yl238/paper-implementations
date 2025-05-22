import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import TSMixer
from plotting import plot_forecasts

if __name__ == "__main__":
    df = pd.read_parquet(
        "data/CAFR_sales_weekly_sales_sales_06_ELECTRICAL HEATING AND COOLING.parquet"
    )
    df["unique_id"] = df["SourceID"] + "_" + df["SiteCode"]

    unique_product_sites = df[
        ["unique_id", "SourceID", "SiteName"]
    ].drop_duplicates()

    df["WeekStart"] = pd.to_datetime(df["WeekStart"])
    df_val = df[(df["WeekStart"] > "2021-06-01") & (df["WeekStart"] <= "2025-01-01")]

    df_total_sales = (
        df_val.groupby("unique_id")
        .agg({"Quantity": "sum"})
        .sort_values(by="Quantity", ascending=False)
    )
    sampled_ids = (
        df_total_sales[df_total_sales["Quantity"] > 10].sample(10000).index.values
    )

    df_val[df_val.unique_id.isin(tuple(sampled_ids))][
        ["SourceID", "SiteCode", "SiteName"]
    ].drop_duplicates().to_csv("test_ids.csv", index=False)

    df_section = df_val[["WeekStart", "Quantity", "unique_id"]]
    f_section = df_section[df_section["unique_id"].isin(tuple(sampled_ids))]

    df_section.reset_index().drop(columns=["index"], inplace=True)
    df_section.columns = ["ds", "y", "unique_id"]

    df_section.groupby("unique_id").agg({"y": "sum"}).sort_values(
        by="y", ascending=False
    ).to_csv("product_sales.csv")

    Y_train_df = df_section[
        df_section.ds < df_section["ds"].values[-52]
    ].reset_index(drop=True)
    Y_test_df = df_section[
        df_section.ds >= df_section["ds"].values[-52]
    ].reset_index(drop=True)

    model = TSMixer(
        h=52,
        input_size=52,
        n_series=10000,
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
    fcst.fit(df=Y_train_df, val_size=52)

    forecasts = fcst.predict(futr_df=Y_test_df)
    forecasts.to_parquet("forecasts.parquet")

    unique_ids = forecasts["unique_id"].unique()
    forecasts["TSMixer-median"][forecasts["TSMixer-median"] < 0] = 0
    forecasts["TSMixer-hi-90"][forecasts["TSMixer-hi-90"] < 0] = 0
    forecasts["TSMixer-lo-90"][forecasts["TSMixer-lo-90"] < 0] = 0

    id = "100110032_1461"
    source = unique_product_sites[unique_product_sites.unique_id == id][
        "SourceID"
    ].values[0]
    site = unique_product_sites[unique_product_sites.unique_id == id][
        "SiteName"
    ].values[0]

    plot_forecasts(
        Y_train_df,
        Y_test_df,
        forecasts,
        id=id,
        sku=source,
        site=site,
        filename=f"forecasts_{id}.png",
    )
