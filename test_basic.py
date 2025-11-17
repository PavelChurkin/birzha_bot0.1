#!/usr/bin/env python3
import sys
sys.path.append('.')
import unittest
from birzha_bot import MoexTradingBot
import pandas as pd
from unittest.mock import patch, MagicMock

class TestMoexTradingBot(unittest.TestCase):
    def setUp(self):
        self.bot = MoexTradingBot()

    def test_get_available_stocks(self):
        stocks = self.bot.get_available_stocks()
        self.assertIsInstance(stocks, dict)
        self.assertGreater(len(stocks), 0)

    def test_search_stocks(self):
        results = self.bot.search_stocks('SBER')
        self.assertIsInstance(results, dict)
        self.assertIn('SBER', results)

    @patch('birzha_bot.requests.get')
    def test_get_stock_data(self, mock_get):
        mock_response = MagicMock()
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = {
            'marketdata': {
                'data': [['TQBR', 100.0, 99.0, 101.0, 98.0, 1000, 100000, 1.0]]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        data = self.bot.get_stock_data('TEST')
        self.assertIsNotNone(data)
        self.assertEqual(data['last'], 100.0)

    def test_calculate_technical_levels(self):
        # Create sample data
        data = {
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        }
        df = pd.DataFrame(data)
        levels = self.bot.calculate_technical_levels(df)
        self.assertIn('rsi', levels)
        self.assertIn('sma_20', levels)

if __name__ == '__main__':
    unittest.main()