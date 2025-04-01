import pandas as pd
from data_fetcher import fetch_data
from indicator_calculator import calculate_indicators
from plotter import plot_data

# Cargar datos
ohlcv = fetch_data()
df = pd.DataFrame(
    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Calcular indicadores
df = calculate_indicators(df)

# Graficar
plot_data(df)
