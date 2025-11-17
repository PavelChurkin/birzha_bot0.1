import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from birzha_bot import MoexTradingBot

class TestMoexTradingBot(unittest.TestCase):
    def setUp(self):
        self.bot = MoexTradingBot()

    def test_calculate_technical_levels(self):
        # Create mock historical data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        data = {
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(105, 115, 50),
            'low': np.random.uniform(95, 105, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.randint(1000, 10000, 50)
        }
        df = pd.DataFrame(data, index=dates)

        levels = self.bot.calculate_technical_levels(df)

        # Check that all expected keys are present
        expected_keys = ['pivot', 'resistance_1', 'support_1', 'supports', 'resistances', 'atr', 'rsi', 'sma_20', 'sma_50']
        for key in expected_keys:
            self.assertIn(key, levels)

        # Check types
        self.assertIsInstance(levels['pivot'], float)
        self.assertIsInstance(levels['rsi'], float)
        self.assertIsInstance(levels['supports'], list)
        self.assertIsInstance(levels['resistances'], list)

    def test_analyze_weekly_trend(self):
        # Create mock weekly data
        dates = pd.date_range(start='2023-01-01', periods=10, freq='W')
        data = {
            'open': [100, 102, 104, 103, 105, 107, 106, 108, 110, 109],
            'high': [105, 107, 109, 108, 110, 112, 111, 113, 115, 114],
            'low': [95, 97, 99, 98, 100, 102, 101, 103, 105, 104],
            'close': [102, 104, 103, 105, 107, 106, 108, 110, 109, 111],
            'volume': [10000] * 10
        }
        df = pd.DataFrame(data, index=dates)

        trend = self.bot.analyze_weekly_trend(df)

        expected_keys = ['trend', 'strength', 'slope', 'weeks_analyzed']
        for key in expected_keys:
            self.assertIn(key, trend)

        self.assertIn(trend['trend'], ['восходящий', 'нисходящий', 'боковой'])
        self.assertIsInstance(trend['strength'], float)
        self.assertGreaterEqual(trend['weeks_analyzed'], 0)

    def test_analyze_intentionality(self):
        # Mock data
        current_data = {
            'last': 105.0,
            'volume': 5000
        }
        hist_data = pd.DataFrame({
            'close': [100, 102, 104, 103, 105, 107],
            'volume': [4000, 4500, 5000, 4800, 5200, 5500],
            'high': [106, 108, 110, 109, 111, 113],
            'low': [98, 100, 102, 101, 103, 105]
        }, index=pd.date_range('2023-01-01', periods=6))

        orderbook = {
            'bids': [{'price': 104.5, 'quantity': 100}, {'price': 104.0, 'quantity': 200}],
            'asks': [{'price': 105.5, 'quantity': 150}, {'price': 106.0, 'quantity': 250}]
        }

        weekly_trend = {'trend': 'восходящий', 'strength': 1.5}
        tech_levels = {'rsi': 65.0, 'sma_20': 103.0, 'sma_50': 102.0}

        signals = self.bot.analyze_intentionality('TEST', current_data, hist_data, orderbook, weekly_trend, tech_levels)

        self.assertIsInstance(signals, list)
        # Check that some signals are generated
        self.assertGreater(len(signals), 0)

    def test_search_stocks(self):
        results = self.bot.search_stocks('Сбер')
        self.assertIsInstance(results, dict)
        # Assuming SBER is in available stocks
        if 'SBER' in self.bot.available_stocks:
            self.assertIn('SBER', results)

    def test_get_available_stocks(self):
        stocks = self.bot.get_available_stocks()
        self.assertIsInstance(stocks, dict)
        self.assertGreater(len(stocks), 0)
        # Check structure
        if stocks:
            sample = next(iter(stocks.values()))
            self.assertIn('name', sample)
            self.assertIn('price', sample)

if __name__ == '__main__':
    unittest.main()