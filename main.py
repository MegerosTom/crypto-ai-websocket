import asyncio
import websockets
import pandas as pd
import json
import os
from prophet import Prophet
from datetime import datetime

CSV_FILE = "data.csv"

async def collect_data():
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            price = float(data["p"])
            quantity = float(data["q"])
            timestamp = datetime.fromtimestamp(data["T"] / 1000.0)

            new_data = pd.DataFrame([[timestamp, price, quantity]], columns=["timestamp", "price", "quantity"])
            if os.path.exists(CSV_FILE):
                new_data.to_csv(CSV_FILE, mode="a", header=False, index=False)
            else:
                new_data.to_csv(CSV_FILE, index=False)

            await asyncio.sleep(1)

def analyze_trend():
    if not os.path.exists(CSV_FILE):
        return
    df = pd.read_csv(CSV_FILE, names=["timestamp", "price", "quantity"], header=0)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").resample("1min").mean().dropna().reset_index()
    df = df.rename(columns={"timestamp": "ds", "price": "y"})

    if len(df) < 10:
        return

    model = Prophet(daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=15, freq='min')
    forecast = model.predict(future)
    print(forecast[["ds", "yhat"]].tail(5))

async def main():
    while True:
        task = asyncio.create_task(collect_data())
        while True:
            await asyncio.sleep(60)
            analyze_trend()

asyncio.run(main())
