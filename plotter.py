import matplotlib.pyplot as plt

def plot_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["close"], label="Precio", color="blue")
    plt.scatter(
        df["timestamp"][df["liquidity_zone_up"]],
        df["high"][df["liquidity_zone_up"]],
        color="red",
        label="Liquidez Arriba",
        marker="^",
    )
    plt.scatter(
        df["timestamp"][df["liquidity_zone_down"]],
        df["low"][df["liquidity_zone_down"]],
        color="green",
        label="Liquidez Abajo",
        marker="v",
    )
    plt.legend()
    plt.show()

