#!/usr/bin/env python3
import sys
import unittest
from unittest.mock import Mock, patch
sys.path.append('.')

from birzha_bot import MoexTradingBot
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestMoexTradingBot(unittest.TestCase):
    
    def setUp(self):
        self.bot = MoexTradingBot()
        # Mock available_stocks
        self.bot.available_stocks = {
            'SBER': {'name': 'Сбербанк', 'full_name': 'Сбербанк России ПАО ао', 'price': 297.44},
            'GAZP': {'name': 'Газпром', 'full_name': 'Газпром ПАО ао', 'price': 118.16}
        }
    
    def test_search_stocks(self):
        results = self.bot.search_stocks("SBER")
        self.assertIn('SBER', results)
        self.assertEqual(results['SBER']['name'], 'Сбербанк')
        
        results = self.bot.search_stocks("газ")
        self.assertIn('GAZP', results)
    
    def test_analyze_weekly_trend(self):
        # Create mock historical data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        data = {
            'open': np.random.uniform(100, 110, 30),
            'high': np.random.uniform(110, 120, 30),
            'low': np.random.uniform(90, 100, 30),
            'close': np.random.uniform(100, 110, 30),
            'volume': np.random.randint(1000, 10000, 30)
        }
        hist_data = pd.DataFrame(data, index=dates)
        
        trend = self.bot.analyze_weekly_trend(hist_data)
        self.assertIn('trend', trend)
        self.assertIn('strength', trend)
        self.assertIsInstance(trend['strength'], float)
    
    def test_calculate_technical_levels(self):
        # Create mock data
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = {
            'open': [100 + i for i in range(20)],
            'high': [105 + i for i in range(20)],
            'low': [95 + i for i in range(20)],
            'close': [102 + i for i in range(20)],
            'volume': [1000] * 20
        }
        hist_data = pd.DataFrame(data, index=dates)
        
        levels = self.bot.calculate_technical_levels(hist_data)
        self.assertIn('pivot', levels)
        self.assertIn('atr', levels)
        self.assertIsInstance(levels['atr'], float)
    
    @patch('birzha_bot.requests.get')
    def test_get_available_stocks(self, mock_get):
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'securities': {
                'data': [
                    ['TEST', 'Test Stock', 'Test Full Name', 100.0]
                ]
            }
        }
        mock_get.return_value = mock_response
        
        bot = MoexTradingBot()
        stocks = bot.get_available_stocks()
        self.assertIn('TEST', stocks)
        self.assertEqual(stocks['TEST']['price'], 100.0)

if __name__ == '__main__':
    unittest.main()