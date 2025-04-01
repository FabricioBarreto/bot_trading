import pandas as pd
import numpy as np
import talib
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndicatorCalculator:
    def __init__(self):
        self.logger = logger
    
    def calculate_basic_indicators(self, df):
        """
        Calcula indicadores básicos para detección de zonas de liquidez
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con indicadores calculados
        """
        try:
            df_result = df.copy()
            
            # Calcular propiedades de velas
            df_result["top_wick"] = df_result["high"] - df_result[["close", "open"]].max(axis=1)
            df_result["bottom_wick"] = df_result[["close", "open"]].min(axis=1) - df_result["low"]
            df_result["body"] = abs(df_result["close"] - df_result["open"])
            
            # Tamaño relativo de mechas
            df_result["long_wick_up"] = df_result["top_wick"] > (df_result["body"] * 1.5)
            df_result["long_wick_down"] = df_result["bottom_wick"] > (df_result["body"] * 1.5)
            
            # Análisis de volumen
            df_result["volume_sma"] = df_result["volume"].rolling(window=20).mean()
            df_result["high_volume"] = df_result["volume"] > df_result["volume_sma"] * 1.5
            
            # Zonas de liquidez potenciales
            df_result["liquidity_zone_up"] = df_result["long_wick_up"] & df_result["high_volume"]
            df_result["liquidity_zone_down"] = df_result["long_wick_down"] & df_result["high_volume"]
            
            # Añadir índices de fuerza
            df_result["liquidity_strength_up"] = 0.0
            df_result["liquidity_strength_down"] = 0.0
            
            # Calcular fuerza de la zona de liquidez basada en tamaño de mecha y volumen
            mask_up = df_result["liquidity_zone_up"]
            if mask_up.any():
                df_result.loc[mask_up, "liquidity_strength_up"] = (
                    df_result.loc[mask_up, "top_wick"] / df_result.loc[mask_up, "body"] *
                    df_result.loc[mask_up, "volume"] / df_result.loc[mask_up, "volume_sma"]
                )
            
            mask_down = df_result["liquidity_zone_down"]
            if mask_down.any():
                df_result.loc[mask_down, "liquidity_strength_down"] = (
                    df_result.loc[mask_down, "bottom_wick"] / df_result.loc[mask_down, "body"] *
                    df_result.loc[mask_down, "volume"] / df_result.loc[mask_down, "volume_sma"]
                )
                
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error al calcular indicadores básicos: {e}")
            raise
    
    def calculate_advanced_indicators(self, df):
        """
        Calcula indicadores técnicos avanzados para ayudar en la toma de decisiones
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con indicadores avanzados
        """
        try:
            df_result = df.copy()
            
            # Asegurar que hay suficientes datos
            if len(df_result) < 50:
                self.logger.warning("Datos insuficientes para calcular indicadores avanzados")
                return df_result
            
            # --- Indicadores de tendencia ---
            
            # Media Móvil Exponencial
            df_result['ema_20'] = talib.EMA(df_result['close'].values, timeperiod=20)
            df_result['ema_50'] = talib.EMA(df_result['close'].values, timeperiod=50)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df_result['close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            df_result['macd'] = macd
            df_result['macd_signal'] = macd_signal
            df_result['macd_hist'] = macd_hist
            
            # Señales de tendencia
            df_result['trend_up'] = (df_result['ema_20'] > df_result['ema_50']) & (df_result['macd'] > df_result['macd_signal'])
            df_result['trend_down'] = (df_result['ema_20'] < df_result['ema_50']) & (df_result['macd'] < df_result['macd_signal'])
            
            # --- Indicadores de momentum ---
            
            # RSI
            df_result['rsi'] = talib.RSI(df_result['close'].values, timeperiod=14)
            
            # Estocástico
            df_result['slowk'], df_result['slowd'] = talib.STOCH(
                df_result['high'].values, 
                df_result['low'].values, 
                df_result['close'].values,
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            
            # Señales de sobrecompra/sobreventa
            df_result['overbought'] = (df_result['rsi'] > 70) | (df_result['slowk'] > 80)
            df_result['oversold'] = (df_result['rsi'] < 30) | (df_result['slowk'] < 20)
            
            # --- Indicadores de volatilidad ---
            
            # Bandas de Bollinger
            df_result['bollinger_upper'], df_result['bollinger_middle'], df_result['bollinger_lower'] = talib.BBANDS(
                df_result['close'].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            
            # ATR (Average True Range)
            df_result['atr'] = talib.ATR(
                df_result['high'].values, 
                df_result['low'].values, 
                df_result['close'].values, 
                timeperiod=14
            )
            
            # Volatilidad relativa (ATR normalizado)
            df_result['volatility'] = df_result['atr'] / df_result['close'] * 100
            df_result['high_volatility'] = df_result['volatility'] > df_result['volatility'].rolling(window=20).mean() * 1.5
            
            # --- Indicadores de volumen avanzados ---
            
            # OBV (On-Balance Volume)
            df_result['obv'] = talib.OBV(df_result['close'].values, df_result['volume'].values)
            
            # Chaikin Money Flow
            df_result['cmf'] = talib.ADOSC(
                df_result['high'].values,
                df_result['low'].values,
                df_result['close'].values,
                df_result['volume'].values,
                fastperiod=3,
                slowperiod=10
            )
            
            # Money Flow Index
            df_result['mfi'] = talib.MFI(
                df_result['high'].values,
                df_result['low'].values,
                df_result['close'].values,
                df_result['volume'].values,
                timeperiod=14
            )
            
            # Señales de divergencia de volumen
            df_result['volume_trend_up'] = (df_result['obv'].diff() > 0) & (df_result['cmf'] > 0) & (df_result['mfi'] > 50)
            df_result['volume_trend_down'] = (df_result['obv'].diff() < 0) & (df_result['cmf'] < 0) & (df_result['mfi'] < 50)
            
            # --- Algoritmos de reconocimiento de patrones ---
            
            # Detección de patrones de velas
            pattern_functions = {
                'doji': talib.CDLDOJI,
                'hammer': talib.CDLHAMMER,
                'engulfing': talib.CDLENGULFING,
                'evening_star': talib.CDLEVENINGSTAR,
                'morning_star': talib.CDLMORNINGSTAR,
                'shooting_star': talib.CDLSHOOTINGSTAR,
                'three_black_crows': talib.CDL3BLACKCROWS,
                'three_white_soldiers': talib.CDL3WHITESOLDIERS
            }
            
            for pattern_name, pattern_func in pattern_functions.items():
                df_result[f'pattern_{pattern_name}'] = pattern_func(
                    df_result['open'].values,
                    df_result['high'].values,
                    df_result['low'].values,
                    df_result['close'].values
                )
            
            # --- Indicadores de liquidez avanzados ---
            
            # Crear variables que capturen la acumulación de liquidez
            df_result['cumulative_liquidity_up'] = df_result['liquidity_strength_up'].rolling(window=10).sum()
            df_result['cumulative_liquidity_down'] = df_result['liquidity_strength_down'].rolling(window=10).sum()
            
            # Interacción con volatilidad
            df_result['liquidity_vol_ratio_up'] = df_result['cumulative_liquidity_up'] / df_result['volatility'].rolling(window=10).mean()
            df_result['liquidity_vol_ratio_down'] = df_result['cumulative_liquidity_down'] / df_result['volatility'].rolling(window=10).mean()
            
            # Probabilidad de captura de liquidez
            df_result['liquidity_capture_prob_up'] = df_result['liquidity_vol_ratio_up'] / df_result['liquidity_vol_ratio_up'].rolling(window=50).max()
            df_result['liquidity_capture_prob_down'] = df_result['liquidity_vol_ratio_down'] / df_result['liquidity_vol_ratio_down'].rolling(window=50).max()
            
            # Llenar NaN con 0
            df_result = df_result.fillna(0)
            
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error al calcular indicadores avanzados: {e}")
            raise
    
    def detect_spoofing_patterns(self, df, order_book_data):
        """
        Detecta patrones de spoofing comparando el libro de órdenes con la acción del precio
        
        Args:
            df: DataFrame con datos OHLCV e indicadores
            order_book_data: Datos del libro de órdenes
            
        Returns:
            DataFrame con alertas de spoofing
        """
        try:
            df_result = df.copy()
            
            if not order_book_data or 'bids' not in order_book_data or 'asks' not in order_book_data:
                self.logger.warning("Datos de libro de órdenes insuficientes para análisis de spoofing")
                df_result['spoofing_alert'] = False
                df_result['spoofing_direction'] = 0
                return df_result
            
            # Extraer órdenes grandes
            bids = order_book_data['bids']
            asks = order_book_data['asks']
            
            # Calcular volumen promedio
            avg_bid_volume = sum([bid[1] for bid in bids]) / len(bids) if bids else 0
            avg_ask_volume = sum([ask[1] for ask in asks]) / len(asks) if asks else 0
            
            # Identificar órdenes grandes (> 5x promedio)
            large_bids = [bid for bid in bids if bid[1] > avg_bid_volume * 5]
            large_asks = [ask for ask in asks if ask[1] > avg_ask_volume * 5]
            
            # Calcular desequilibrio entre compra y venta
            bid_volume = sum([bid[1] for bid in large_bids])
            ask_volume = sum([ask[1] for ask in large_asks])
            
            imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
            reverse_imbalance_ratio = ask_volume / bid_volume if bid_volume > 0 else float('inf')
            
            # Detectar posible spoofing
            spoofing_bid = imbalance_ratio > 3 and df_result['trend_down'].iloc[-1]
            spoofing_ask = reverse_imbalance_ratio > 3 and df_result['trend_up'].iloc[-1]
            
            df_result['spoofing_alert'] = spoofing_bid or spoofing_ask
            df_result['spoofing_direction'] = -1 if spoofing_bid else (1 if spoofing_ask else 0)
            
            # Detalles adicionales
            df_result['bid_volume_ratio'] = bid_volume / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.5
            df_result['ask_volume_ratio'] = ask_volume / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.5
            
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error al detectar patrones de spoofing: {e}")
            raise
    
    def identify_market_structure(self, df, window_size=10):
        """
        Identifica la estructura del mercado (swing highs/lows, consolidaciones, etc.)
        
        Args:
            df: DataFrame con datos OHLCV e indicadores
            window_size: Tamaño de ventana para análisis de estructura
            
        Returns:
            DataFrame con información de estructura de mercado
        """
        try:
            df_result = df.copy()
            
            # Identificar swing highs y lows
            df_result['swing_high'] = False
            df_result['swing_low'] = False
            
            for i in range(window_size, len(df_result) - window_size):
                # Swing high: máximo local rodeado de valores menores
                if all(df_result['high'].iloc[i] > df_result['high'].iloc[i-j] for j in range(1, window_size+1)) and \
                   all(df_result['high'].iloc[i] > df_result['high'].iloc[i+j] for j in range(1, window_size+1)):
                    df_result.loc[df_result.index[i], 'swing_high'] = True
                
                # Swing low: mínimo local rodeado de valores mayores
                if all(df_result['low'].iloc[i] < df_result['low'].iloc[i-j] for j in range(1, window_size+1)) and \
                   all(df_result['low'].iloc[i] < df_result['low'].iloc[i+j] for j in range(1, window_size+1)):
                    df_result.loc[df_result.index[i], 'swing_low'] = True
            
            # Detectar consolidaciones (rango estrecho de precios)
            rolling_range = (df_result['high'].rolling(window=window_size).max() - 
                             df_result['low'].rolling(window=window_size).min()) / df_result['close']
            df_result['consolidation'] = rolling_range < rolling_range.rolling(window=window_size*3).mean() * 0.5
            
            # Detectar expansiones (rango amplio de precios)
            df_result['expansion'] = rolling_range > rolling_range.rolling(window=window_size*3).mean() * 1.5
            
            # Análisis de tendencia basado en swing points
            df_result['higher_high'] = False
            df_result['higher_low'] = False
            df_result['lower_high'] = False
            df_result['lower_low'] = False
            
            # Encontrar índices de swing points
            swing_high_indices = df_result.index[df_result['swing_high']].tolist()
            swing_low_indices = df_result.index[df_result['swing_low']].tolist()
            
            # Analizar secuencia de swing highs
            for i in range(1, len(swing_high_indices)):
                curr_idx = swing_high_indices[i]
                prev_idx = swing_high_indices[i-1]
                
                if df_result.loc[curr_idx, 'high'] > df_result.loc[prev_idx, 'high']:
                    df_result.loc[curr_idx, 'higher_high'] = True
                else:
                    df_result.loc[curr_idx, 'lower_high'] = True
            
            # Analizar secuencia de swing lows
            for i in range(1, len(swing_low_indices)):
                curr_idx = swing_low_indices[i]
                prev_idx = swing_low_indices[i-1]
                
                if df_result.loc[curr_idx, 'low'] > df_result.loc[prev_idx, 'low']:
                    df_result.loc[curr_idx, 'higher_low'] = True
                else:
                    df_result.loc[curr_idx, 'lower_low'] = True
            
            # Definir estructura de mercado
            df_result['market_structure'] = 'neutral'
            
            # Estructura alcista: higher highs y higher lows
            uptrend_mask = (df_result['higher_high'] | df_result['higher_low']) & ~(df_result['lower_high'] | df_result['lower_low'])
            df_result.loc[uptrend_mask, 'market_structure'] = 'bullish'
            
            # Estructura bajista: lower highs y lower lows
            downtrend_mask = (df_result['lower_high'] | df_result['lower_low']) & ~(df_result['higher_high'] | df_result['higher_low'])
            df_result.loc[downtrend_mask, 'market_structure'] = 'bearish'
            
            # Estructura de rango: mixed highs y lows o consolidación
            range_mask = ((df_result['higher_high'] | df_result['higher_low']) & (df_result['lower_high'] | df_result['lower_low'])) | df_result['consolidation']
            df_result.loc[range_mask, 'market_structure'] = 'range'
            
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error al identificar estructura de mercado: {e}")
            raise
    
    def generate_trading_signals(self, df, order_book_data=None, risk_factor=1.0):
        """
        Genera señales de trading basadas en todos los indicadores y análisis
        
        Args:
            df: DataFrame con datos OHLCV e indicadores
            order_book_data: Datos del libro de órdenes (opcional)
            risk_factor: Factor de ajuste de riesgo (0.5 = conservador, 1.0 = normal, 2.0 = agresivo)
            
        Returns:
            DataFrame con señales de trading
        """
        try:
            # Paso 1: Calcular indicadores básicos
            df_result = self.calculate_basic_indicators(df)
            
            # Paso 2: Calcular indicadores avanzados
            df_result = self.calculate_advanced_indicators(df_result)
            
            # Paso 3: Detección de spoofing si hay datos de libro de órdenes
            if order_book_data:
                df_result = self.detect_spoofing_patterns(df_result, order_book_data)
            
            # Paso 4: Identificar estructura del mercado
            df_result = self.identify_market_structure(df_result)
            
            # Paso 5: Generar señales finales de trading
            
            # Inicializar columnas de señal
            df_result['entry_signal'] = 0  # 0: nada, 1: compra, -1: venta
            df_result['signal_strength'] = 0.0  # 0.0-1.0
            df_result['stop_loss'] = 0.0
            df_result['take_profit'] = 0.0
            df_result['risk_reward_ratio'] = 0.0
            
            # Obtener última fila (datos actuales)
            last_row = df_result.iloc[-1]
            
            # Agregado de factores de señal
            signal_factors = {
                'trend': 0,
                'oscillator': 0,
                'volume': 0,
                'liquidity': 0,
                'pattern': 0,
                'spoofing': 0,
                'market_structure': 0
            }
            
            # Factor de tendencia
            if last_row['trend_up']:
                signal_factors['trend'] = 1
            elif last_row['trend_down']:
                signal_factors['trend'] = -1
            
            # Factor de oscilador
            if last_row['oversold'] and last_row['rsi'] < 30:
                signal_factors['oscillator'] = 1
            elif last_row['overbought'] and last_row['rsi'] > 70:
                signal_factors['oscillator'] = -1
            
            # Factor de volumen
            if last_row['volume_trend_up']:
                signal_factors['volume'] = 1
            elif last_row['volume_trend_down']:
                signal_factors['volume'] = -1
            
            # Factor de liquidez
            if last_row['liquidity_capture_prob_up'] > 0.7:
                signal_factors['liquidity'] = 1
            elif last_row['liquidity_capture_prob_down'] > 0.7:
                signal_factors['liquidity'] = -1
            
            # Factor de patrón
            pattern_score = 0
            bullish_patterns = ['pattern_hammer', 'pattern_morning_star', 'pattern_three_white_soldiers']
            bearish_patterns = ['pattern_shooting_star', 'pattern_evening_star', 'pattern_three_black_crows']
            
            for pattern in bullish_patterns:
                if last_row[pattern] > 0:
                    pattern_score += 1
            
            for pattern in bearish_patterns:
                if last_row[pattern] > 0:
                    pattern_score -= 1
            
            signal_factors['pattern'] = np.sign(pattern_score)
            
            # Factor de spoofing
            if 'spoofing_direction' in last_row:
                signal_factors['spoofing'] = last_row['spoofing_direction']
            
            # Factor de estructura de mercado
            if last_row['market_structure'] == 'bullish':
                signal_factors['market_structure'] = 1
            elif last_row['market_structure'] == 'bearish':
                signal_factors['market_structure'] = -1
            
            # Calcular señal combinada
            weights = {
                'trend': 0.2,
                'oscillator': 0.15,
                'volume': 0.15,
                'liquidity': 0.2,
                'pattern': 0.1,
                'spoofing': 0.1,
                'market_structure': 0.1
            }
            
            combined_signal = sum(signal_factors[k] * weights[k] for k in signal_factors)
            
            # Aplicar factor de riesgo
            combined_signal *= risk_factor
            
            # Determinar señal final
            if combined_signal > 0.3:
                df_result.loc[df_result.index[-1], 'entry_signal'] = 1
                df_result.loc[df_result.index[-1], 'signal_strength'] = min(abs(combined_signal), 1.0)
            elif combined_signal < -0.3:
                df_result.loc[df_result.index[-1], 'entry_signal'] = -1
                df_result.loc[df_result.index[-1], 'signal_strength'] = min(abs(combined_signal), 1.0)
            
            # Calcular stop loss y take profit
            if df_result.loc[df_result.index[-1], 'entry_signal'] != 0:
                # ATR para determinar volatilidad
                atr_value = last_row['atr']
                
                if df_result.loc[df_result.index[-1], 'entry_signal'] == 1:  # Compra
                    df_result.loc[df_result.index[-1], 'stop_loss'] = last_row['close'] - (atr_value * 2)
                    df_result.loc[df_result.index[-1], 'take_profit'] = last_row['close'] + (atr_value * 3)
                else:  # Venta
                    df_result.loc[df_result.index[-1], 'stop_loss'] = last_row['close'] + (atr_value * 2)
                    df_result.loc[df_result.index[-1], 'take_profit'] = last_row['close'] - (atr_value * 3)
                
                # Calcular ratio riesgo/recompensa
                risk = abs(last_row['close'] - df_result.loc[df_result.index[-1], 'stop_loss'])
                reward = abs(last_row['close'] - df_result.loc[df_result.index[-1], 'take_profit'])
                df_result.loc[df_result.index[-1], 'risk_reward_ratio'] = reward / risk if risk > 0 else 0
            
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error al generar señales de trading: {e}")
            raise

# Función para compatibilidad con código antiguo
def calculate_indicators(df):
    calculator = IndicatorCalculator()
    result_df = calculator.calculate_basic_indicators(df)
    return result_df