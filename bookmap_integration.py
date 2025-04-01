import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BookmapClient:
    def __init__(self, api_key, base_url="https://api.bookmap.com/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def fetch_market_depth(self, symbol, depth=20):
        """Obtiene datos de profundidad de mercado para un símbolo específico"""
        endpoint = f"{self.base_url}/depth/{symbol}"
        params = {"depth": depth}

        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener datos de profundidad: {e}")
            return None

    def fetch_historical_depth(self, symbol, start_time, end_time, interval="1m"):
        """Obtiene datos históricos de profundidad para un símbolo"""
        endpoint = f"{self.base_url}/historical/depth/{symbol}"

        if isinstance(start_time, datetime):
            start_time = int(start_time.timestamp() * 1000)
        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp() * 1000)

        params = {"startTime": start_time, "endTime": end_time, "interval": interval}

        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener datos históricos de profundidad: {e}")
            return None

    def fetch_large_orders(self, symbol, threshold_volume=100):
        """Obtiene órdenes grandes actuales en el libro de órdenes"""
        endpoint = f"{self.base_url}/orders/large/{symbol}"
        params = {"thresholdVolume": threshold_volume}

        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener órdenes grandes: {e}")
            return None


class LiquidityAnalyzer:
    def __init__(self, bookmap_client):
        self.bookmap_client = bookmap_client
        # Horarios de apertura de mercados importantes (UTC)
        self.market_hours = {
            "US_Open": {"hour": 13, "minute": 30},  # 9:30 AM ET
            "US_Close": {"hour": 20, "minute": 0},  # 4:00 PM ET
            "London_Open": {"hour": 8, "minute": 0},
            "London_Close": {"hour": 16, "minute": 30},
            "Tokyo_Open": {"hour": 0, "minute": 0},
            "Tokyo_Close": {"hour": 6, "minute": 0},
        }

    def detect_liquidity_zones(
        self, market_depth_data, price_history, threshold_ratio=2.0
    ):
        """
        Identifica zonas de liquidez basadas en la profundidad del mercado y el historial de precios

        Args:
            market_depth_data: Datos de profundidad de mercado de Bookmap
            price_history: DataFrame con historial de precios OHLCV
            threshold_ratio: Ratio para considerar volumen significativo respecto al promedio

        Returns:
            DataFrame con zonas de liquidez detectadas
        """
        if not market_depth_data:
            return pd.DataFrame()

        # Calcular el volumen promedio en el historial reciente
        avg_volume = price_history["volume"].rolling(20).mean().iloc[-1]

        liquidity_zones = []

        # Analizar órdenes de compra (bids)
        for bid in market_depth_data.get("bids", []):
            price = bid.get("price")
            volume = bid.get("volume")

            # Considerar órdenes con volumen significativamente mayor al promedio
            if volume > avg_volume * threshold_ratio:
                liquidity_zones.append(
                    {
                        "price": price,
                        "volume": volume,
                        "side": "bid",
                        "type": "liquidity_zone",
                        "confidence": min(
                            volume / (avg_volume * threshold_ratio) * 100, 100
                        ),
                    }
                )

        # Analizar órdenes de venta (asks)
        for ask in market_depth_data.get("asks", []):
            price = ask.get("price")
            volume = ask.get("volume")

            if volume > avg_volume * threshold_ratio:
                liquidity_zones.append(
                    {
                        "price": price,
                        "volume": volume,
                        "side": "ask",
                        "type": "liquidity_zone",
                        "confidence": min(
                            volume / (avg_volume * threshold_ratio) * 100, 100
                        ),
                    }
                )

        # Verificar zonas de liquidez históricas
        for index, row in price_history.iterrows():
            # Zonas de liquidación históricas basadas en mechas largas con volumen alto
            if row["long_wick_up"] and row["high_volume"]:
                price_level = row["high"]
                # Verificar si ya existe una zona cercana
                exists = False
                for zone in liquidity_zones:
                    if (
                        abs(zone["price"] - price_level) / price_level < 0.005
                    ):  # Dentro del 0.5%
                        exists = True
                        zone["confidence"] = min(
                            zone["confidence"] + 10, 100
                        )  # Aumentar confianza
                        break

                if not exists:
                    liquidity_zones.append(
                        {
                            "price": price_level,
                            "volume": row["volume"],
                            "side": "ask",
                            "type": "historical_liquidity",
                            "confidence": 75,  # Confianza base para zonas históricas
                        }
                    )

            # Similar para mechas largas hacia abajo
            if row["long_wick_down"] and row["high_volume"]:
                price_level = row["low"]
                exists = False
                for zone in liquidity_zones:
                    if abs(zone["price"] - price_level) / price_level < 0.005:
                        exists = True
                        zone["confidence"] = min(zone["confidence"] + 10, 100)
                        break

                if not exists:
                    liquidity_zones.append(
                        {
                            "price": price_level,
                            "volume": row["volume"],
                            "side": "bid",
                            "type": "historical_liquidity",
                            "confidence": 75,
                        }
                    )

        # Convertir a DataFrame para facilitar manipulación
        return pd.DataFrame(liquidity_zones)

    def detect_spoofing(
        self, historical_depth_data, time_window=5, order_life_threshold=2
    ):
        """
        Detecta posible spoofing analizando órdenes que aparecen y desaparecen rápidamente

        Args:
            historical_depth_data: Serie temporal de datos de profundidad
            time_window: Número de intervalos para análisis
            order_life_threshold: Umbral de tiempo de vida de orden para considerar spoofing

        Returns:
            Lista de posibles alertas de spoofing
        """
        if not historical_depth_data or len(historical_depth_data) < time_window:
            return []

        spoofing_alerts = []

        # Rastrear órdenes grandes que aparecen y desaparecen
        order_tracker = {}

        for timestamp_idx, depth_snapshot in enumerate(historical_depth_data):
            timestamp = depth_snapshot.get("timestamp")

            # Procesar órdenes de compra (bids)
            for bid in depth_snapshot.get("bids", []):
                order_id = f"bid_{bid.get('price')}_{bid.get('volume')}"

                # Si es una orden grande
                if bid.get("volume") > 5 * sum(
                    [b.get("volume", 0) for b in depth_snapshot.get("bids", [])]
                ) / len(depth_snapshot.get("bids", [])):
                    # Si es nueva, registrarla
                    if order_id not in order_tracker:
                        order_tracker[order_id] = {
                            "first_seen": timestamp_idx,
                            "last_seen": timestamp_idx,
                            "price": bid.get("price"),
                            "volume": bid.get("volume"),
                            "side": "bid",
                        }
                    else:
                        # Actualizar última vez vista
                        order_tracker[order_id]["last_seen"] = timestamp_idx

            # Hacer lo mismo para órdenes de venta (asks)
            for ask in depth_snapshot.get("asks", []):
                order_id = f"ask_{ask.get('price')}_{ask.get('volume')}"

                if ask.get("volume") > 5 * sum(
                    [a.get("volume", 0) for a in depth_snapshot.get("asks", [])]
                ) / len(depth_snapshot.get("asks", [])):
                    if order_id not in order_tracker:
                        order_tracker[order_id] = {
                            "first_seen": timestamp_idx,
                            "last_seen": timestamp_idx,
                            "price": ask.get("price"),
                            "volume": ask.get("volume"),
                            "side": "ask",
                        }
                    else:
                        order_tracker[order_id]["last_seen"] = timestamp_idx

        # Analizar órdenes rastreadas para detectar spoofing
        for order_id, order_info in order_tracker.items():
            order_lifetime = order_info["last_seen"] - order_info["first_seen"]

            # Si la orden fue vista por poco tiempo y era grande
            if 0 < order_lifetime <= order_life_threshold:
                spoofing_alerts.append(
                    {
                        "price": order_info["price"],
                        "volume": order_info["volume"],
                        "side": order_info["side"],
                        "type": "potential_spoofing",
                        "confidence": min(
                            (order_life_threshold - order_lifetime) * 25, 90
                        ),  # Más corta la vida, mayor confianza
                    }
                )

        return spoofing_alerts

    def check_market_hours(self, timestamp):
        """
        Verifica si el timestamp está cerca de aperturas o cierres de mercados importantes

        Args:
            timestamp: datetime a verificar

        Returns:
            dict: Información sobre eventos de mercado cercanos
        """
        market_events = {}

        # Convertir a UTC para comparación
        dt_utc = timestamp
        if not isinstance(timestamp, datetime):
            try:
                dt_utc = datetime.fromtimestamp(timestamp / 1000)
            except:
                dt_utc = datetime.now()

        # Verificar proximidad a eventos de mercado
        for market, time_info in self.market_hours.items():
            market_time = datetime(
                dt_utc.year,
                dt_utc.month,
                dt_utc.day,
                time_info["hour"],
                time_info["minute"],
            )

            # Si es el mismo día pero ya pasó, ajustar para mañana
            if dt_utc.time() > market_time.time():
                market_time = market_time + timedelta(days=1)

            # Calcular tiempo hasta evento
            time_diff = (market_time - dt_utc).total_seconds() / 60  # en minutos

            # Si está a menos de 60 minutos
            if 0 <= time_diff <= 60:
                market_events[market] = {
                    "event_type": (
                        "upcoming_open" if "Open" in market else "upcoming_close"
                    ),
                    "minutes_until": time_diff,
                }
            # Si acaba de ocurrir (en los últimos 30 minutos)
            elif -30 <= time_diff < 0:
                market_events[market] = {
                    "event_type": "recent_open" if "Open" in market else "recent_close",
                    "minutes_since": -time_diff,
                }

        return market_events

    def analyze_price_action(
        self, df, liquidity_zones, spoofing_alerts, current_time=None
    ):
        """
        Analiza la acción del precio en relación a zonas de liquidez y spoofing

        Args:
            df: DataFrame con datos OHLCV y indicadores
            liquidity_zones: DataFrame con zonas de liquidez
            spoofing_alerts: Lista de alertas de spoofing
            current_time: Timestamp actual

        Returns:
            DataFrame enriquecido con análisis
        """
        df_result = df.copy()

        # Añadir columnas para análisis
        df_result["price_to_liq_distance"] = 0
        df_result["liq_zone_confidence"] = 0
        df_result["nearest_liq_side"] = ""
        df_result["spoofing_alert"] = False
        df_result["market_event"] = ""
        df_result["entry_signal"] = 0  # 0: neutral, 1: compra, -1: venta

        if current_time is None:
            current_time = datetime.now()

        # Verificar eventos de mercado cercanos
        market_events = self.check_market_hours(current_time)
        market_event_str = ""

        for market, event_info in market_events.items():
            if event_info["event_type"] == "upcoming_open":
                market_event_str += (
                    f"{market} abre en {int(event_info['minutes_until'])} min. "
                )
            elif event_info["event_type"] == "upcoming_close":
                market_event_str += (
                    f"{market} cierra en {int(event_info['minutes_until'])} min. "
                )
            elif event_info["event_type"] == "recent_open":
                market_event_str += (
                    f"{market} abrió hace {int(event_info['minutes_since'])} min. "
                )
            elif event_info["event_type"] == "recent_close":
                market_event_str += (
                    f"{market} cerró hace {int(event_info['minutes_since'])} min. "
                )

        df_result["market_event"] = market_event_str

        # Obtener último precio
        last_price = df_result["close"].iloc[-1]

        # Analizar distancia a zonas de liquidez
        if not liquidity_zones.empty:
            # Calcular distancia a cada zona
            liquidity_zones["distance"] = (
                abs(liquidity_zones["price"] - last_price) / last_price * 100
            )  # en porcentaje

            # Ordenar por distancia
            sorted_zones = liquidity_zones.sort_values("distance")

            if not sorted_zones.empty:
                nearest_zone = sorted_zones.iloc[0]
                df_result.loc[df_result.index[-1], "price_to_liq_distance"] = (
                    nearest_zone["distance"]
                )
                df_result.loc[df_result.index[-1], "liq_zone_confidence"] = (
                    nearest_zone["confidence"]
                )
                df_result.loc[df_result.index[-1], "nearest_liq_side"] = nearest_zone[
                    "side"
                ]

                # Generar señal basada en proximidad y confianza en zona de liquidez
                if nearest_zone["distance"] < 1.0 and nearest_zone["confidence"] > 70:
                    if nearest_zone["side"] == "ask":
                        df_result.loc[df_result.index[-1], "entry_signal"] = (
                            1  # señal de compra
                        )
                    else:
                        df_result.loc[df_result.index[-1], "entry_signal"] = (
                            -1
                        )  # señal de venta

        # Incorporar alertas de spoofing
        if spoofing_alerts:
            df_result.loc[df_result.index[-1], "spoofing_alert"] = True

            # Modificar señal si hay spoofing en dirección contraria
            for alert in spoofing_alerts:
                # Si la alerta es de alta confianza, puede anular señal
                if alert.get("confidence", 0) > 80:
                    # Spoofing en bids (compras) suele indicar movimiento bajista
                    if (
                        alert.get("side") == "bid"
                        and df_result.loc[df_result.index[-1], "entry_signal"] == 1
                    ):
                        df_result.loc[df_result.index[-1], "entry_signal"] = 0
                    # Spoofing en asks (ventas) suele indicar movimiento alcista
                    elif (
                        alert.get("side") == "ask"
                        and df_result.loc[df_result.index[-1], "entry_signal"] == -1
                    ):
                        df_result.loc[df_result.index[-1], "entry_signal"] = 0

        # Considerar eventos de mercado para ajustar señales
        if market_event_str:
            # Mayor volatilidad esperada cerca de aperturas de mercados importantes
            if "US_Open" in market_event_str or "London_Open" in market_event_str:
                # Reforzar señales existentes
                if df_result.loc[df_result.index[-1], "entry_signal"] != 0:
                    # Reforzar señal actual multiplicando por 1.5
                    signal_val = df_result.loc[df_result.index[-1], "entry_signal"]
                    df_result.loc[df_result.index[-1], "entry_signal"] = (
                        signal_val * 1.5
                    )

        return df_result


def integrate_bookmap_data(df, api_key):
    """
    Integra datos de Bookmap con el análisis técnico existente

    Args:
        df: DataFrame con datos OHLCV y indicadores
        api_key: API key de Bookmap

    Returns:
        DataFrame enriquecido con datos de Bookmap
    """
    # Inicializar cliente y analizador
    bookmap_client = BookmapClient(api_key)
    liquidity_analyzer = LiquidityAnalyzer(bookmap_client)

    # Obtener el símbolo en formato Bookmap (ajustar según documentación)
    symbol = "BTCUSD"

    try:
        # Obtener profundidad de mercado actual
        market_depth = bookmap_client.fetch_market_depth(symbol)

        # Obtener datos históricos de profundidad para detectar spoofing
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        historical_depth = bookmap_client.fetch_historical_depth(
            symbol, start_time, end_time
        )

        # Detectar zonas de liquidez
        liquidity_zones = liquidity_analyzer.detect_liquidity_zones(market_depth, df)

        # Detectar posible spoofing
        spoofing_alerts = liquidity_analyzer.detect_spoofing(historical_depth)

        # Analizar acción del precio
        enriched_df = liquidity_analyzer.analyze_price_action(
            df, liquidity_zones, spoofing_alerts
        )

        return enriched_df, liquidity_zones, spoofing_alerts

    except Exception as e:
        logger.error(f"Error en la integración de Bookmap: {e}")
        return df, pd.DataFrame(), []
