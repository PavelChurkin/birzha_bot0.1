#!/usr/bin/env python3
import sys
sys.path.append('.')
from birzha_bot import MoexTradingBot

def test_analysis():
    bot = MoexTradingBot()
    analysis = bot.generate_trading_ranges('SBER')
    bot.print_analysis(analysis)

if __name__ == "__main__":
    test_analysis()