import os
import re
import json
import math
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify
import requests
import logging
from dataclasses import dataclass
import hashlib

app = Flask(__name__)

# ========= CONFIGURATION =========
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '').strip()
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '').strip()
ALERT_MODE = os.getenv('ALERT_MODE', 'enhanced').strip()  # basic, enhanced, full
# =================================

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TimeframeConfig:
    """Configuration for each timeframe"""
    multiplier: float
    risk_multiplier: float
    sl_multiplier: float
    tp_multiplier: float
    min_confidence: float
    valid_for_hours: int

class TimeframeCalculator:
    """Calculate optimal parameters based on timeframe"""
    
    # Timeframe configuration mapping
    TIMEFRAME_CONFIGS = {
        # Intraday timeframes
        '1m': TimeframeConfig(multiplier=0.3, risk_multiplier=0.5, sl_multiplier=0.7, tp_multiplier=0.8, min_confidence=0.7, valid_for_hours=0.5),
        '2m': TimeframeConfig(multiplier=0.4, risk_multiplier=0.6, sl_multiplier=0.75, tp_multiplier=0.85, min_confidence=0.65, valid_for_hours=1),
        '3m': TimeframeConfig(multiplier=0.45, risk_multiplier=0.65, sl_multiplier=0.78, tp_multiplier=0.88, min_confidence=0.62, valid_for_hours=1.5),
        '4m': TimeframeConfig(multiplier=0.48, risk_multiplier=0.68, sl_multiplier=0.79, tp_multiplier=0.89, min_confidence=0.61, valid_for_hours=1.5),
        '5m': TimeframeConfig(multiplier=0.5, risk_multiplier=0.7, sl_multiplier=0.8, tp_multiplier=0.9, min_confidence=0.6, valid_for_hours=2),
        '15m': TimeframeConfig(multiplier=0.7, risk_multiplier=0.8, sl_multiplier=0.85, tp_multiplier=1.0, min_confidence=0.55, valid_for_hours=4),
        '30m': TimeframeConfig(multiplier=0.8, risk_multiplier=0.9, sl_multiplier=0.9, tp_multiplier=1.1, min_confidence=0.5, valid_for_hours=8),
        
        # Hourly timeframes
        '1H': TimeframeConfig(multiplier=1.0, risk_multiplier=1.0, sl_multiplier=1.0, tp_multiplier=1.0, min_confidence=0.45, valid_for_hours=24),
        '2H': TimeframeConfig(multiplier=1.2, risk_multiplier=1.1, sl_multiplier=1.1, tp_multiplier=1.2, min_confidence=0.4, valid_for_hours=48),
        '4H': TimeframeConfig(multiplier=1.5, risk_multiplier=1.2, sl_multiplier=1.2, tp_multiplier=1.3, min_confidence=0.35, valid_for_hours=72),
        
        # Daily and above
        '1D': TimeframeConfig(multiplier=2.0, risk_multiplier=1.5, sl_multiplier=1.5, tp_multiplier=1.5, min_confidence=0.3, valid_for_hours=168),
        '1W': TimeframeConfig(multiplier=3.0, risk_multiplier=2.0, sl_multiplier=2.0, tp_multiplier=2.0, min_confidence=0.25, valid_for_hours=336),
        '1M': TimeframeConfig(multiplier=4.0, risk_multiplier=2.5, sl_multiplier=2.5, tp_multiplier=2.5, min_confidence=0.2, valid_for_hours=720),
    }
    
    # Base parameters for different instrument types (for 1H timeframe)
    BASE_PARAMS = {
        'FOREX': {
            'base_sl_pips': 20,
            'base_tp_pips': 40,
            'pip_value': 0.0001,
            'min_position_size': 0.01,
            'price_decimals': 5
        },
        'INDICES': {
            'base_sl_points': 50,
            'base_tp_points': 100,
            'point_value': 1.0,
            'min_position_size': 0.1,
            'price_decimals': 2
        },
        'COMMODITIES': {
            'base_sl_percent': 1.0,
            'base_tp_percent': 2.0,
            'min_position_size': 0.01,
            'price_decimals': 2
        },
        'CRYPTO': {
            'base_sl_percent': 2.0,
            'base_tp_percent': 4.0,
            'min_position_size': 0.001,
            'price_decimals': 2
        }
    }
    
    @classmethod
    def get_timeframe_config(cls, timeframe_str: str) -> TimeframeConfig:
        """Get configuration for a specific timeframe"""
        # Normalize timeframe string
        tf = timeframe_str.upper()
        if tf.endswith('MIN'):
            tf = tf.replace('MIN', 'm')
        elif tf.endswith('H'):
            tf = tf.replace('H', 'H')
        elif tf.endswith('D'):
            tf = tf.replace('D', 'D')
        elif tf.endswith('W'):
            tf = tf.replace('W', 'W')
        elif tf.endswith('M'):
            tf = tf.replace('M', 'M')
        
        # Map common aliases
        tf_map = {
            '1': '1m', '2': '2m', '3': '3m', '4': '4m', '5': '5m',
            '10': '10m', '15': '15m', '30': '30m',
            '60': '1H', '120': '2H', '240': '4H', '360': '6H', '480': '8H', '720': '12H',
            'D': '1D', '1D': '1D', '1440': '1D',
            'W': '1W', '1W': '1W', '10080': '1W',
            'M': '1M', '1M': '1M'
        }
        
        tf = tf_map.get(tf, tf)
        
        # Get config or return default for 1H
        return cls.TIMEFRAME_CONFIGS.get(tf, cls.TIMEFRAME_CONFIGS['1H'])
    
    def calculate_signal_parameters(self, entry_price: float, direction: str, 
                                   instrument_type: str, timeframe: str,
                                   market_data: dict = None) -> dict:
        """Calculate all signal parameters based on timeframe"""
        
        # Get timeframe configuration
        tf_config = self.get_timeframe_config(timeframe)
        
        # Get base parameters for instrument
        base_params = self.BASE_PARAMS.get(instrument_type, self.BASE_PARAMS['FOREX'])
        
        # Calculate adjusted parameters
        if instrument_type == 'FOREX':
            # Forex: Use pips with timeframe multiplier
            sl_pips = base_params['base_sl_pips'] * tf_config.sl_multiplier
            tp_pips = base_params['base_tp_pips'] * tf_config.tp_multiplier
            
            sl_distance = sl_pips * base_params['pip_value']
            tp_distance = tp_pips * base_params['pip_value']
            
        elif instrument_type == 'INDICES':
            # Indices: Use points with timeframe multiplier
            sl_points = base_params['base_sl_points'] * tf_config.sl_multiplier
            tp_points = base_params['base_tp_points'] * tf_config.tp_multiplier
            
            sl_distance = sl_points * base_params['point_value']
            tp_distance = tp_points * base_params['point_value']
            
        else:
            # Commodities & Crypto: Use percentages with timeframe multiplier
            sl_percent = base_params['base_sl_percent'] * tf_config.sl_multiplier
            tp_percent = base_params['base_tp_percent'] * tf_config.tp_multiplier
            
            sl_distance = entry_price * (sl_percent / 100)
            tp_distance = entry_price * (tp_percent / 100)
        
        # Calculate actual price levels
        if direction in ['LONG', 'BUY']:  # Support both old and new naming
            sl_price = entry_price - sl_distance
            tp1_price = entry_price + (tp_distance * 0.5)
            tp2_price = entry_price + tp_distance
            tp3_price = entry_price + (tp_distance * 1.5)
            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        else:  # SHORT or SELL
            sl_price = entry_price + sl_distance
            tp1_price = entry_price - (tp_distance * 0.5)
            tp2_price = entry_price - tp_distance
            tp3_price = entry_price - (tp_distance * 1.5)
            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        
        # Calculate position size based on risk
        risk_per_trade = 1.0 * tf_config.risk_multiplier  # Base 1% risk
        position_size = self.calculate_position_size(
            entry_price, sl_distance, risk_per_trade, 
            base_params.get('min_position_size', 0.01)
        )
        
        # Calculate signal validity
        valid_until = datetime.now(timezone.utc) + timedelta(hours=tf_config.valid_for_hours)
        
        # Format numbers
        price_decimals = base_params['price_decimals']
        
        return {
            'stop_loss': round(sl_price, price_decimals),
            'take_profit_1': round(tp1_price, price_decimals),
            'take_profit_2': round(tp2_price, price_decimals),
            'take_profit_3': round(tp3_price, price_decimals),
            'risk_reward_ratio': round(rr_ratio, 2),
            'position_size': round(position_size, 4),
            'valid_until': valid_until.isoformat(),
            'timeframe_multiplier': tf_config.multiplier,
            'min_confidence': tf_config.min_confidence,
            'price_decimals': price_decimals
        }
    
    def calculate_position_size(self, entry_price: float, sl_distance: float, 
                               risk_percent: float, min_size: float) -> float:
        """Calculate position size based on risk management"""
        if sl_distance == 0:
            return min_size
        
        # Calculate risk per unit
        risk_per_unit = sl_distance
        
        # Calculate quantity
        quantity = (risk_percent / 100) / (risk_per_unit / entry_price)
        
        # Ensure minimum size
        return max(min_size, quantity)
    
    def analyze_timeframe_quality(self, timeframe: str, instrument_type: str) -> dict:
        """Analyze if timeframe is suitable for the instrument"""
        tf_config = self.get_timeframe_config(timeframe)
        
        # Define optimal timeframes for each instrument
        optimal_tfs = {
            'FOREX': ['15m', '1H', '4H', '1D'],
            'INDICES': ['15m', '1H', '4H', '1D'],
            'COMMODITIES': ['1H', '4H', '1D'],
            'CRYPTO': ['5m', '15m', '1H', '4H']
        }
        
        is_optimal = timeframe in optimal_tfs.get(instrument_type, [])
        
        return {
            'is_optimal': is_optimal,
            'recommended_timeframes': optimal_tfs.get(instrument_type, []),
            'volatility_factor': tf_config.multiplier,
            'suitability_score': 0.8 if is_optimal else 0.5
        }

class SignalValidator:
    """Validate trading signals with timeframe awareness"""
    
    def __init__(self):
        self.timeframe_calc = TimeframeCalculator()
        self.signal_history = {}
    
    def validate_enhanced_signal(self, signal_data: dict) -> dict:
        """Validate signal with all parameters"""
        
        validation = {
            'is_valid': True,
            'confidence': 1.0,
            'warnings': [],
            'recommendations': [],
            'rejection_reasons': []
        }
        
        try:
            # Extract data
            pair = signal_data.get('pair', '')
            action = signal_data.get('action', '')
            price = signal_data.get('price', 0)
            timeframe = signal_data.get('timeframe', '1H')
            instrument_type = signal_data.get('instrument_type', 'FOREX')
            
            # Normalize action for validation
            action_normalized = self._normalize_action(action)
            
            # Check if signal is expired
            if 'entry_time' in signal_data:
                entry_time = datetime.fromtimestamp(int(signal_data['entry_time']) / 1000, timezone.utc)
                tf_config = self.timeframe_calc.get_timeframe_config(timeframe)
                max_age = timedelta(hours=tf_config.valid_for_hours)
                
                if datetime.now(timezone.utc) - entry_time > max_age:
                    validation['is_valid'] = False
                    validation['rejection_reasons'].append(f"Signal expired (older than {tf_config.valid_for_hours} hours)")
            
            # Check duplicate signal
            signal_hash = self._create_signal_hash(signal_data)
            if signal_hash in self.signal_history:
                time_diff = datetime.now(timezone.utc) - self.signal_history[signal_hash]
                if time_diff < timedelta(minutes=5):
                    validation['is_valid'] = False
                    validation['rejection_reasons'].append("Duplicate signal (received within 5 minutes)")
            
            # Validate price
            if price <= 0:
                validation['is_valid'] = False
                validation['rejection_reasons'].append("Invalid price")
            
            # Validate timeframe suitability
            tf_analysis = self.timeframe_calc.analyze_timeframe_quality(timeframe, instrument_type)
            if not tf_analysis['is_optimal']:
                validation['warnings'].append(f"Timeframe {timeframe} may not be optimal for {instrument_type}")
                validation['confidence'] *= 0.8
            
            # Check market conditions if provided
            if 'adx_val' in signal_data and 'ranging' in signal_data:
                adx_val = signal_data['adx_val']
                is_ranging = signal_data['ranging']
                
                if not is_ranging and action_normalized in ['LONG', 'SHORT']:
                    # Mean reversion strategy works best in ranging markets
                    validation['warnings'].append("Market is trending (ADX > 25)")
                    validation['confidence'] *= 0.6
            
            # Check volatility
            if 'volatility' in signal_data:
                volatility = signal_data['volatility']
                if volatility > 2.0:  # High volatility
                    validation['warnings'].append(f"High volatility detected ({volatility:.1f}%)")
                    validation['confidence'] *= 0.7
            
            # Check Bollinger Band position
            if all(k in signal_data for k in ['bb_upper', 'bb_middle', 'bb_lower']):
                upper = signal_data['bb_upper']
                lower = signal_data['bb_lower']
                current_price = float(price)
                
                # Check if price is at band extremes for mean reversion
                if action_normalized == 'LONG' and current_price > lower * 1.01:
                    validation['warnings'].append("Price not at lower Bollinger Band")
                    validation['confidence'] *= 0.9
                elif action_normalized == 'SHORT' and current_price < upper * 0.99:
                    validation['warnings'].append("Price not at upper Bollinger Band")
                    validation['confidence'] *= 0.9
            
            # Apply timeframe-specific minimum confidence
            tf_config = self.timeframe_calc.get_timeframe_config(timeframe)
            if validation['confidence'] < tf_config.min_confidence:
                validation['is_valid'] = False
                validation['rejection_reasons'].append(f"Confidence too low ({validation['confidence']:.1%} < {tf_config.min_confidence:.1%})")
            
            # Store signal in history
            if validation['is_valid']:
                self.signal_history[signal_hash] = datetime.now(timezone.utc)
                # Clean old signals
                self._clean_old_signals()
            
            validation['confidence'] = round(validation['confidence'], 3)
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation['is_valid'] = False
            validation['rejection_reasons'].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _normalize_action(self, action: str) -> str:
        """Normalize action to standard format"""
        if not action:
            return ''
        
        action = str(action).upper().strip()
        
        # Map to consistent naming
        action_map = {
            'BUY': 'LONG',
            'BUYING': 'LONG',
            'LONG': 'LONG',
            'SELL': 'SHORT',
            'SELLING': 'SHORT',
            'SHORT': 'SHORT',
            'EXIT_LONG': 'EXIT_LONG',
            'EXIT_SHORT': 'EXIT_SHORT',
            'CLOSE_LONG': 'EXIT_LONG',
            'CLOSE_SHORT': 'EXIT_SHORT'
        }
        
        return action_map.get(action, action)
    
    def _create_signal_hash(self, signal_data: dict) -> str:
        """Create unique hash for signal"""
        hash_string = f"{signal_data.get('pair', '')}{signal_data.get('action', '')}{signal_data.get('price', '')}"
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _clean_old_signals(self):
        """Remove old signals from history"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self.signal_history = {
            k: v for k, v in self.signal_history.items() 
            if v > cutoff_time
        }

def parse_timeframe(tf_str: str) -> str:
    """Parse and normalize timeframe string"""
    if not tf_str or tf_str == 'N/A':
        return '1H'  # Default
    
    tf_str = str(tf_str).upper().strip()
    
    # Map common formats
    tf_map = {
        '1': '1m', '2': '2m', '3': '3m', '4': '4m', '5': '5m',
        '10': '10m', '15': '15m', '30': '30m',
        '60': '1H', '120': '2H', '240': '4H', '360': '6H', '480': '8H', '720': '12H',
        'D': '1D', '1D': '1D', '1440': '1D',
        'W': '1W', '1W': '1W', '10080': '1W',
        'M': '1M', '1M': '1M'
    }
    
    return tf_map.get(tf_str, tf_str)

def detect_instrument_type(pair: str) -> str:
    """Detect instrument type from pair name"""
    pair = str(pair).upper()
    
    # Indices
    indices = ['GER30', 'NAS100', 'SPX500', 'US30', 'UK100', 'JPN225', 'DXY', 'NQ', 'ES', 'YM']
    if any(index in pair for index in indices):
        return 'INDICES'
    
    # Commodities
    commodities = ['XAU', 'GOLD', 'XAG', 'SILVER', 'OIL', 'BRENT', 'WTI', 'XPT', 'PLATINUM', 'CL', 'GC']
    if any(comm in pair for comm in commodities):
        return 'COMMODITIES'
    
    # Crypto
    cryptos = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT', 'BNB', 'MATIC', 'AVAX']
    if any(crypto in pair for crypto in cryptos):
        return 'CRYPTO'
    
    return 'FOREX'  # Default

def normalize_action_name(action: str) -> str:
    """Normalize action name to standard format"""
    if not action:
        return ''
    
    action = str(action).upper().strip()
    
    # Map to consistent naming - FIXED VERSION
    if action in ['BUY', 'BUYING', 'LONG']:
        return 'LONG'
    elif action in ['SELL', 'SELLING', 'SHORT']:
        return 'SHORT'
    elif 'EXIT_LONG' in action or 'CLOSE_LONG' in action:
        return 'EXIT_LONG'
    elif 'EXIT_SHORT' in action or 'CLOSE_SHORT' in action:
        return 'EXIT_SHORT'
    else:
        return action

def format_telegram_message(signal_data: dict, validation: dict, 
                           signal_params: dict) -> str:
    """Format enhanced Telegram message - FIXED VERSION"""
    
    action = signal_data.get('action', '')
    pair = signal_data.get('pair', '')
    price = signal_data.get('price', '')
    timeframe = signal_data.get('timeframe', '')
    
    # Normalize action for display
    normalized_action = normalize_action_name(action)
    
    # Determine emoji and title - FIXED LOGIC
    if 'EXIT' in normalized_action:
        emoji = "🔴"
        title = "EXIT SIGNAL"
        direction_display = normalized_action.replace('EXIT_', '').title()
    elif normalized_action == 'LONG':
        emoji = "🟢"
        title = "LONG ENTRY"
        direction_display = "LONG"
    elif normalized_action == 'SHORT':
        emoji = "🔵"
        title = "SHORT ENTRY"
        direction_display = "SHORT"
    else:
        emoji = "⚪"
        title = "TRADING SIGNAL"
        direction_display = normalized_action
    
    message = f"{emoji} *{title}* {emoji}\n\n"
    
    # Basic info - FIXED: Use "Direction" instead of "Action"
    message += f"*Instrument:* `{pair}`\n"
    message += f"*Type:* `{signal_data.get('instrument_type', 'N/A')}`\n"
    message += f"*Direction:* `{direction_display}`\n"
    message += f"*Entry:* `{price}`\n"
    message += f"*Reason:* `{signal_data.get('reason', 'N/A')}`\n"
    message += f"*Timeframe:* `{timeframe}`\n"
    
    # Timeframe analysis
    tf_calc = TimeframeCalculator()
    tf_config = tf_calc.get_timeframe_config(timeframe)
    tf_analysis = tf_calc.analyze_timeframe_quality(timeframe, signal_data.get('instrument_type', 'FOREX'))
    
    message += f"\n*⏰ Timeframe Analysis:*\n"
    message += f"• Multiplier: `{tf_config.multiplier}x`\n"
    message += f"• Optimal: `{'✅' if tf_analysis['is_optimal'] else '⚠️'}`\n"
    message += f"• Valid for: `{tf_config.valid_for_hours}h`\n"
    
    # Signal parameters
    decimals = signal_params.get('price_decimals', 2)
    if signal_params:
        message += f"\n*📊 Signal Parameters:*\n"
        message += f"• Stop Loss: `{signal_params.get('stop_loss', 'N/A')}`\n"
        message += f"• Take Profit 1: `{signal_params.get('take_profit_1', 'N/A')}`\n"
        message += f"• Take Profit 2: `{signal_params.get('take_profit_2', 'N/A')}`\n"
        message += f"• Take Profit 3: `{signal_params.get('take_profit_3', 'N/A')}`\n"
        message += f"• Risk/Reward: `{signal_params.get('risk_reward_ratio', 'N/A')}`\n"
        message += f"• Suggested Position Size: `{signal_params.get('position_size', 'N/A')}`\n"
    
    # Strategy-specific info
    message += f"\n*🛠 Strategy Info:*\n"
    message += f"• ATR: `{signal_data.get('atr', 0):.4f}`\n"
    message += f"• Adaptive Multiplier: `{signal_data.get('adaptive_mult', 0):.1f}`\n"
    message += f"• Avg Recent Profit: `{signal_data.get('avg_profit', 'N/A')}`\n"
    if 'sl_price' in signal_data:
        message += f"• Strategy SL: `{round(signal_data.get('sl_price', 0), decimals)}`\n"
        message += f"• Strategy TP: `{round(signal_data.get('tp_price', 0), decimals)}`\n"
        message += f"• Strategy Position Size: `{signal_data.get('position_size', 'N/A')}`\n"
    
    # Validation info
    message += f"\n*✅ Validation:*\n"
    message += f"• Confidence: `{validation.get('confidence', 0):.1%}`\n"
    message += f"• Status: `{'VALID ✅' if validation.get('is_valid', False) else 'REJECTED ❌'}`\n"
    
    if validation.get('warnings'):
        warnings = validation['warnings'][:3]  # Show only first 3 warnings
        message += f"• Warnings: `{', '.join(warnings)}`\n"
    
    # Market context if available
    if 'adx_val' in signal_data:
        message += f"\n*📈 Market Context:*\n"
        message += f"• ADX: `{signal_data.get('adx_val', 0):.1f}`\n"
        message += f"• Regime: `{'RANGING' if signal_data.get('ranging', True) else 'TRENDING'}`\n"
    
    if 'volatility' in signal_data:
        message += f"• Volatility: `{signal_data.get('volatility', 0):.1f}%`\n"
    
    if 'bb_upper' in signal_data:
        message += f"• BB Upper: `{round(signal_data.get('bb_upper', 0), decimals)}`\n"
        message += f"• BB Middle: `{round(signal_data.get('bb_middle', 0), decimals)}`\n"
        message += f"• BB Lower: `{round(signal_data.get('bb_lower', 0), decimals)}`\n"
    
    # Timestamps
    current_time = datetime.now(timezone.utc)
    message += f"\n*🕒 Timestamps:*\n"
    message += f"• Signal Time: `{current_time.strftime('%H:%M UTC')}`\n"
    
    if 'valid_until' in signal_params:
        valid_time = datetime.fromisoformat(signal_params['valid_until'].replace('Z', '+00:00'))
        message += f"• Valid Until: `{valid_time.strftime('%H:%M UTC')}`\n"
    
    return message

@app.route('/webhook', methods=['POST', 'GET'])
def handle_webhook():
    """Enhanced webhook handler with timeframe-aware calculations"""
    
    logger.info("=" * 70)
    logger.info("ENHANCED WEBHOOK RECEIVED")
    
    if request.method == 'GET':
        return jsonify({
            "status": "ready",
            "service": "Timeframe-Aware Trading Bot",
            "version": "7.0",
            "mode": ALERT_MODE
        }), 200
    
    try:
        # Get raw data
        raw_data = request.get_data(as_text=True).strip()
        logger.info(f"Received data: {raw_data[:200]}...")
        
        # Initialize processors
        validator = SignalValidator()
        timeframe_calc = TimeframeCalculator()
        
        # Parse based on alert mode
        signal_data = {}
        
        if ALERT_MODE == "enhanced":
            try:
                # Try to parse as JSON first
                signal_data = json.loads(raw_data)
                logger.info("Parsed as JSON alert")
            except json.JSONDecodeError:
                # Attempt to fix common JSON issues, like extra quotes or missing braces
                fixed_data = raw_data.replace('true"', 'true').replace('false"', 'false').replace(',}', '}')
                if not raw_data.endswith('}'):
                    fixed_data += '}'
                try:
                    signal_data = json.loads(fixed_data)
                    logger.info("Parsed after fixing JSON")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON even after fix: {e}")
                    return jsonify({"status": "parse_error", "message": str(e)}), 200
        else:
            # Basic mode - simple parsing
            parts = raw_data.split()
            if len(parts) >= 4:
                signal_data = {
                    'pair': parts[0],
                    'action': parts[1],
                    'price': float(parts[3]) if len(parts) > 3 else 0,
                    'timeframe': parts[-1] if 'on' in raw_data else '1H'
                }
        
        # Ensure required fields
        if not signal_data or 'pair' not in signal_data or 'action' not in signal_data:
            logger.error("Invalid signal data received")
            return jsonify({"status": "invalid_data"}), 200
        
        # Convert types (since now all are strings from Pine)
        signal_data['price'] = float(signal_data['price'])
        signal_data['adx_val'] = float(signal_data['adx_val'])
        signal_data['ranging'] = str(signal_data['ranging']).lower() == 'true'
        signal_data['volatility'] = float(signal_data['volatility'])
        signal_data['bb_upper'] = float(signal_data['bb_upper'])
        signal_data['bb_middle'] = float(signal_data['bb_middle'])
        signal_data['bb_lower'] = float(signal_data['bb_lower'])
        signal_data['entry_time'] = int(signal_data['entry_time'])
        if 'atr' in signal_data:
            signal_data['atr'] = float(signal_data['atr'])
        if 'adaptive_mult' in signal_data:
            signal_data['adaptive_mult'] = float(signal_data['adaptive_mult'])
        if 'avg_profit' in signal_data:
            signal_data['avg_profit'] = float(signal_data['avg_profit'])
        if 'sl_price' in signal_data:
            signal_data['sl_price'] = float(signal_data['sl_price'])
        if 'tp_price' in signal_data:
            signal_data['tp_price'] = float(signal_data['tp_price'])
        if 'position_size' in signal_data:
            signal_data['position_size'] = float(signal_data['position_size'])
        
        # Normalize timeframe
        if 'timeframe' in signal_data:
            signal_data['timeframe'] = parse_timeframe(signal_data['timeframe'])
        else:
            signal_data['timeframe'] = '1H'  # Default
        
        # Determine instrument type
        signal_data['instrument_type'] = detect_instrument_type(signal_data['pair'])
        
        # FIXED: Normalize action name to ensure consistency
        signal_data['action'] = normalize_action_name(signal_data['action'])
        
        logger.info(f"Signal: {signal_data['pair']} {signal_data['action']} @ {signal_data.get('price', 0)} TF:{signal_data['timeframe']}")
        
        # Validate signal
        validation = validator.validate_enhanced_signal(signal_data)
        logger.info(f"Validation result: {validation}")
        
        # Calculate signal parameters based on timeframe
        if validation['is_valid'] and signal_data.get('price', 0) > 0:
            try:
                signal_params = timeframe_calc.calculate_signal_parameters(
                    entry_price=float(signal_data['price']),
                    direction=signal_data['action'],
                    instrument_type=signal_data['instrument_type'],
                    timeframe=signal_data['timeframe'],
                    market_data=signal_data
                )
            except Exception as e:
                logger.error(f"Error calculating signal parameters: {e}")
                signal_params = {}
        else:
            signal_params = {}
        
        # Send to Telegram if valid and credentials available
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            if validation['is_valid'] and signal_params:
                # Format message
                message = format_telegram_message(signal_data, validation, signal_params)
                
                # Send to Telegram
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                payload = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                    "disable_notification": False
                }
                
                logger.info("Sending signal to Telegram...")
                response = requests.post(url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    logger.info("✅ Signal sent successfully")
                    
                    # Log the successful signal
                    log_signal(signal_data, validation, signal_params)
                    
                    return jsonify({
                        "status": "success",
                        "validation": validation,
                        "parameters": signal_params,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }), 200
                else:
                    logger.error(f"❌ Telegram error: {response.status_code} - {response.text}")
                    return jsonify({"status": "telegram_error", "details": response.text}), 200
            else:
                # Log rejected signal
                logger.warning(f"Signal rejected: {validation.get('rejection_reasons', [])}")
                
                # Optionally send rejection alert (for debugging)
                if os.getenv('SEND_REJECTIONS', 'false').lower() == 'true':
                    reject_message = f"❌ *SIGNAL REJECTED*\n\n"
                    reject_message += f"*Pair:* `{signal_data.get('pair', '')}`\n"
                    reject_message += f"*Reason:* `{', '.join(validation.get('rejection_reasons', ['Unknown']))}`\n"
                    reject_message += f"*Time:* `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`"
                    
                    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                    payload = {
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": reject_message,
                        "parse_mode": "Markdown"
                    }
                    requests.post(url, json=payload, timeout=5)
                
                return jsonify({
                    "status": "rejected",
                    "validation": validation,
                    "reasons": validation.get('rejection_reasons', [])
                }), 200
        else:
            logger.error("Missing Telegram credentials")
            return jsonify({"status": "no_credentials"}), 200
            
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 200
    
    finally:
        logger.info("=" * 70)

def log_signal(signal_data: dict, validation: dict, parameters: dict):
    """Log successful signal for analysis"""
    log_entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'signal': signal_data,
        'validation': validation,
        'parameters': parameters
    }
    
    # In production, save to database or file
    logger.info(f"Signal logged: {signal_data.get('pair')} {signal_data.get('action')} "
                f"(Confidence: {validation.get('confidence', 0):.1%})")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "operational",
        "service": "Timeframe-Aware Trading Bot v7.1",
        "version": "7.1",
        "mode": ALERT_MODE,
        "telegram_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200

@app.route('/test-signal', methods=['POST'])
def test_signal():
    """Test endpoint for signal processing"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Initialize processors
        validator = SignalValidator()
        timeframe_calc = TimeframeCalculator()
        
        # Parse data
        signal_data = data.copy()
        
        # Add required fields if missing
        if 'timeframe' not in signal_data:
            signal_data['timeframe'] = '1H'
        
        if 'instrument_type' not in signal_data and 'pair' in signal_data:
            signal_data['instrument_type'] = detect_instrument_type(signal_data['pair'])
        
        # Normalize action
        signal_data['action'] = normalize_action_name(signal_data.get('action', ''))
        
        # Validate
        validation = validator.validate_enhanced_signal(signal_data)
        
        # Calculate parameters
        if signal_data.get('price', 0) > 0:
            signal_params = timeframe_calc.calculate_signal_parameters(
                entry_price=float(signal_data.get('price', 0)),
                direction=signal_data.get('action', ''),
                instrument_type=signal_data.get('instrument_type', 'FOREX'),
                timeframe=signal_data.get('timeframe', '1H'),
                market_data=signal_data
            )
        else:
            signal_params = {}
        
        # Format message for preview
        message = format_telegram_message(signal_data, validation, signal_params)
        
        return jsonify({
            "signal_data": signal_data,
            "validation": validation,
            "parameters": signal_params,
            "message_preview": message[:500] + "..." if len(message) > 500 else message
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check configuration"""
    return jsonify({
        "telegram_token_set": bool(TELEGRAM_TOKEN),
        "telegram_token_length": len(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else 0,
        "telegram_chat_id": TELEGRAM_CHAT_ID,
        "alert_mode": ALERT_MODE,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "telegram_api_test": "Run /test-telegram to check"
    }), 200

@app.route('/test-telegram', methods=['GET'])
def test_telegram():
    """Test Telegram connection"""
    try:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return jsonify({"error": "Telegram not configured"}), 400
        
        # Test getMe
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            return jsonify({
                "status": "error",
                "telegram_api": "failed",
                "error": response.text
            }), 400
        
        # Test sendMessage
        test_message = f"✅ Telegram Test\nTime: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": test_message,
            "parse_mode": "Markdown"
        }
        
        send_response = requests.post(url, json=payload, timeout=5)
        
        return jsonify({
            "status": "success",
            "telegram_api": "connected",
            "bot_info": response.json(),
            "message_sent": send_response.status_code == 200,
            "message_response": send_response.json() if send_response.status_code == 200 else send_response.text
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logger.info(f"🚀 Starting Timeframe-Aware Trading Bot v7.1 on port {port}")
    logger.info(f"🔧 Alert Mode: {ALERT_MODE}")
    logger.info(f"🤖 Telegram configured: {bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)}")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)