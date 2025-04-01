import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, exchange_id="binance", retry_attempts=3, retry_delay=2):
        self.exchange_id = exchange_id
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.exchange = self._initialize_exchange()

    def _initialize_exchange(self):
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class(
                {
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "future",  # Para futuros de BTCUSD
                    },
                }
            )
            return exchange
        except Exception as e:
            logger.error(f"Error al inicializar exchange {self.exchange_id}: {e}")
            raise

    def fetch_ohlcv(self, symbol="BTC/USDT", timeframe="1h", limit=500, since=None):
        """
        Obtiene datos OHLCV de un exchange con reintentos

        Args:
            symbol: Par de trading
            timeframe: Intervalo de tiempo
            limit: Número máximo de candles
            since: Timestamp desde donde empezar (ms)

        Returns:
            DataFrame con datos OHLCV
        """
        for attempt in range(self.retry_attempts):
            try:
                # Obtener datos crudos
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=limit, since=since
                )

                # Convertir a DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )

                # Convertir timestamp a datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                return df

            except Exception as e:
                logger.warning(
                    f"Intento {attempt+1}/{self.retry_attempts} fallido: {e}"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Error al obtener datos OHLCV: {e}")
                    raise

    def fetch_order_book(self, symbol="BTC/USDT", limit=100):
        """
        Obtiene el libro de órdenes actual

        Args:
            symbol: Par de trading
            limit: Profundidad del libro

        Returns:
            Dict con el libro de órdenes
        """
        for attempt in range(self.retry_attempts):
            try:
                order_book = self.exchange.fetch_order_book(symbol, limit)
                return order_book
            except Exception as e:
                logger.warning(
                    f"Intento {attempt+1}/{self.retry_attempts} fallido: {e}"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Error al obtener libro de órdenes: {e}")
                    raise

    def fetch_multiple_timeframes(
        self, symbol="BTC/USDT", timeframes=["1h", "4h", "1d"], limit=100
    ):
        """
        Obtiene datos OHLCV para múltiples timeframes

        Args:
            symbol: Par de trading
            timeframes: Lista de timeframes a obtener
            limit: Número de candles por timeframe

        Returns:
            Dict con DataFrames para cada timeframe
        """
        result = {}

        for tf in timeframes:
            try:
                result[tf] = self.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
                logger.info(f"Obtenidos datos para {symbol} en timeframe {tf}")
            except Exception as e:
                logger.error(f"Error al obtener datos para timeframe {tf}: {e}")
                result[tf] = None

        return result

    def get_market_hours_status(self):
        """
        Verifica el estado actual de los principales mercados financieros

        Returns:
            Dict con información sobre mercados abiertos/cerrados
        """
        now_utc = datetime.utcnow()
        weekday = now_utc.weekday()  # 0-6 (lunes-domingo)

        # Definir horarios de mercado (UTC)
        markets = {
            "US": {
                "open_hour": 13,  # 9:30 AM ET aproximado
                "close_hour": 20,  # 4:00 PM ET aproximado
                "open_weekdays": [0, 1, 2, 3, 4],  # Lunes a viernes
            },
            "London": {
                "open_hour": 8,
                "close_hour": 16,
                "open_weekdays": [0, 1, 2, 3, 4],
            },
            "Tokyo": {
                "open_hour": 0,
                "close_hour": 6,
                "open_weekdays": [0, 1, 2, 3, 4],
            },
            "Crypto": {
                "open_hour": 0,
                "close_hour": 23,
                "open_weekdays": [0, 1, 2, 3, 4, 5, 6],  # Siempre abierto
            },
        }

        result = {}

        for market_name, market_info in markets.items():
            is_open_day = weekday in market_info["open_weekdays"]
            current_hour = now_utc.hour

            is_open = (
                is_open_day
                and market_info["open_hour"]
                <= current_hour
                <= market_info["close_hour"]
            )

            # Para mercados que abrirán pronto
            minutes_to_open = None
            if is_open_day and current_hour < market_info["open_hour"]:
                minutes_to_open = (
                    market_info["open_hour"] - current_hour
                ) * 60 - now_utc.minute

            # Para mercados que cerrarán pronto
            minutes_to_close = None
            if is_open and current_hour == market_info["close_hour"]:
                minutes_to_close = 60 - now_utc.minute

            result[market_name] = {
                "is_open": is_open,
                "minutes_to_open": minutes_to_open,
                "minutes_to_close": minutes_to_close,
            }

        return result


# Función para compatibilidad con código antiguo
def fetch_data(symbol="BTC/USDT", timeframe="1h", limit=500):
    fetcher = DataFetcher()
    df = fetcher.fetch_ohlcv(symbol, timeframe, limit)
    return df.values.tolist()
