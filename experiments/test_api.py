#!/usr/bin/env python3
import sys
sys.path.append('.')
from birzha_bot import MoexTradingBot

def test_api():
    bot = MoexTradingBot()
    print("Available stocks:", len(bot.available_stocks))
    
    # Test with SBER
    data = bot.get_stock_data('SBER')
    print("Stock data for SBER:", data)
    
    hist = bot.get_historical_data('SBER', days=7)
    print("Historical data shape:", hist.shape if hist is not None else None)
    
    orderbook = bot.get_orderbook('SBER')
    print("Orderbook bids/asks:", len(orderbook['bids']) if orderbook else 0, len(orderbook['asks']) if orderbook else 0)

if __name__ == "__main__":
    test_api()