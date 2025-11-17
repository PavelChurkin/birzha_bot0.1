#!/usr/bin/env python3
import sys
sys.path.append('.')

from birzha_bot import MoexTradingBot

def test_basic():
    bot = MoexTradingBot()
    print("Available stocks:", len(bot.available_stocks))

    # Test search
    results = bot.search_stocks("SBER")
    print("Search SBER:", results)

    # Test get stock data
    data = bot.get_stock_data("SBER")
    print("SBER data:", data)

    # Test historical data
    hist = bot.get_historical_data("SBER", days=7)
    print("SBER hist shape:", hist.shape if hist is not None else None)

    # Test orderbook
    ob = bot.get_orderbook("SBER")
    print("SBER orderbook:", ob is not None)

    # Test analysis
    analysis = bot.generate_trading_ranges("SBER")
    print("Analysis done:", analysis is not None)

if __name__ == "__main__":
    test_basic()