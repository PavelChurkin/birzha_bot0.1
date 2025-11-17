import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional

class MoexTradingBot:
    def __init__(self):
        self.base_url = "https://iss.moex.com/iss"
        self.available_stocks = self.get_available_stocks()
    
    def get_available_stocks(self) -> Dict:
        """Получение списка доступных акций с MOEX"""
        url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities.json"
        params = {
            'iss.only': 'securities',
            'securities.columns': 'SECID,SHORTNAME,SECNAME,PREVPRICE'   # PREVPRICE - цена предыдущего дня
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            securities = data['securities']['data']
            
            stocks = {}
            for sec in securities:
                if sec[3] and sec[3] > 0:  # Фильтр по акциям с ценой
                    stocks[sec[0]] = {
                        'name': sec[1],
                        'full_name': sec[2],
                        'price': sec[3]
                    }
            return stocks
            
        except Exception as e:
            print(f"Ошибка получения списка акций: {e}")
            return {}
    
    def search_stocks(self, query: str) -> Dict:
        """Поиск акций по названию или тикеру"""
        results = {}
        query = query.lower()
        
        for ticker, info in self.available_stocks.items():
            if (query in ticker.lower() or 
                query in info['name'].lower() or 
                query in info['full_name'].lower()):
                results[ticker] = info
                
        return results
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Получение текущих данных по акции"""
        url = f"{self.base_url}/engines/stock/markets/shares/securities/{symbol}.json"
        params = {
            'iss.only': 'marketdata',
            'marketdata.columns': 'LAST,OPEN,HIGH,LOW,VOLTODAY,VALTODAY,LASTTOPREVPRICE'     # VOLTODAY - объем торгов за день, VALTODAY - стоимость торгов
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            marketdata = data['marketdata']['data']
            
            if marketdata and marketdata[0][0] is not None:
                return {
                    'symbol': symbol,
                    'last': marketdata[0][0],
                    'open': marketdata[0][1],
                    'high': marketdata[0][2],
                    'low': marketdata[0][3],
                    'volume': marketdata[0][4],
                    'value': marketdata[0][5],
                    'change': marketdata[0][6],
                    'timestamp': datetime.now()
                }
            return None
            
        except Exception as e:
            print(f"Ошибка получения данных для {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Получение исторических данных"""
        url = f"{self.base_url}/engines/stock/markets/shares/securities/{symbol}/candles.json"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            'from': start_date.strftime('%Y-%m-%d'),
            'till': end_date.strftime('%Y-%m-%d'),
            'interval': 24,
            'iss.only': 'candles',
            'candles.columns': 'open,high,low,close,volume,begin'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            candles = data['candles']['data']
            
            if not candles:
                return None
                
            df = pd.DataFrame(candles, columns=['open', 'high', 'low', 'close', 'volume', 'begin'])
            df['begin'] = pd.to_datetime(df['begin'])
            df.set_index('begin', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Ошибка получения исторических данных для {symbol}: {e}")
            return None
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Получение стакана заявок"""
        url = f"{self.base_url}/engines/stock/markets/shares/securities/{symbol}/orderbook.json"
        
        try:
            response = requests.get(url)
            data = response.json()
            orderbook = data['orderbook']['data']
            
            bids = []
            asks = []
            
            for item in orderbook:
                if item[0] == 'B' and item[1] and item[2]:
                    bids.append({'price': item[1], 'quantity': item[2]})
                elif item[0] == 'S' and item[1] and item[2]:
                    asks.append({'price': item[1], 'quantity': item[2]})
            
            return {
                'bids': sorted(bids, key=lambda x: x['price'], reverse=True)[:10],
                'asks': sorted(asks, key=lambda x: x['price'])[:10]
            }
            
        except Exception as e:
            print(f"Ошибка получения стакана для {symbol}: {e}")
            return None
    
    def calculate_technical_levels(self, df: pd.DataFrame) -> Dict:
        """Расчет технических уровней"""
        if df.empty:
            return {}
        
        # Pivot Points
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        
        # Поддержка/сопротивление по экстремумам
        window = min(10, len(df) // 3)
        supports = []
        resistances = []
        
        for i in range(window, len(df)-window):
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
                supports.append(df['low'].iloc[i])
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
                resistances.append(df['high'].iloc[i])
        
        supports = sorted(list(set([round(x, 2) for x in supports if not np.isnan(x)])))
        resistances = sorted(list(set([round(x, 2) for x in resistances if not np.isnan(x)])))
        
        # ATR (волатильность)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean().iloc[-1] if len(true_range) > 14 else true_range.mean()
        
        return {
            'pivot': round(pivot, 2),
            'resistance_1': round(r1, 2),
            'support_1': round(s1, 2),
            'supports': supports[-3:] if supports else [],
            'resistances': resistances[-3:] if resistances else [],
            'atr': round(atr, 2)
        }
    
    def analyze_weekly_trend(self, hist_data: pd.DataFrame) -> Dict:
        """Анализ недельного тренда"""
        if hist_data.empty or len(hist_data) < 7:
            return {'trend': 'недостаточно данных', 'strength': 0}
        
        # Пересчет на недельные данные
        weekly = hist_data.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(weekly) < 2:
            return {'trend': 'недостаточно данных', 'strength': 0.0}
        
        # Расчет тренда по последним 4 неделям
        recent_weeks = weekly.tail(4)
        closes = recent_weeks['close']
        
        # Линейный тренд
        x = np.arange(len(closes))
        slope, _ = np.polyfit(x, closes, 1)
        
        # Определение тренда
        if slope > 0.001 * closes.iloc[0]:
            trend = 'восходящий'
            strength = min(abs(slope) / closes.iloc[0] * 100, 5)  # сила в %
        elif slope < -0.001 * closes.iloc[0]:
            trend = 'нисходящий'
            strength = min(abs(slope) / closes.iloc[0] * 100, 5)
        else:
            trend = 'боковой'
            strength = 0
        
        return {
            'trend': trend,
            'strength': round(strength, 2),
            'slope': round(slope, 4),
            'weeks_analyzed': len(recent_weeks)
        }
    
    def analyze_intentionality(self, symbol: str, current_data: Dict, hist_data: pd.DataFrame, orderbook: Optional[Dict], weekly_trend: Dict) -> List:
        """Анализ интенциональности"""
        signals = []
        
        # Анализ объема
        if not hist_data.empty and current_data.get('volume'):
            volume_avg = hist_data['volume'].tail(20).mean()
            current_volume = current_data['volume']
            
            if current_volume > volume_avg * 2:
                signals.append("💥 СИЛЬНЫЙ ОБЪЕМ")
            elif current_volume > volume_avg * 1.5:
                signals.append("📈 ПОВЫШЕННЫЙ ОБЪЕМ")
        
        # Анализ стакана
        if orderbook:
            bid_volume = sum([bid['quantity'] for bid in orderbook['bids']])
            ask_volume = sum([ask['quantity'] for ask in orderbook['asks']])
            total_levels = len(orderbook['bids']) + len(orderbook['asks'])
            density = (bid_volume + ask_volume) / max(total_levels, 1) if total_levels > 0 else 0
            
            if bid_volume + ask_volume > 0:
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                if imbalance > 0.3:
                    signals.append("🟢 ПРЕОБЛАДАЮТ ПОКУПКИ")
                elif imbalance < -0.3:
                    signals.append("🔴 ПРЕОБЛАДАЮТ ПРОДАЖИ")
            
            # Анализ плотности заявок
            if density > 10000:  # Высокая плотность
                signals.append("📊 ВЫСОКАЯ ПЛОТНОСТЬ ЗАЯВОК")
            elif density < 1000:  # Низкая плотность
                signals.append("📉 НИЗКАЯ ПЛОТНОСТЬ ЗАЯВОК")
            
            # Анализ спреда
            if orderbook['bids'] and orderbook['asks']:
                best_bid = max(bid['price'] for bid in orderbook['bids'])
                best_ask = min(ask['price'] for ask in orderbook['asks'])
                spread = best_ask - best_bid
                spread_pct = spread / best_bid * 100
                if spread_pct < 0.1:
                    signals.append("💰 УЗКИЙ СПРЕД (высокая ликвидность)")
                elif spread_pct > 1:
                    signals.append("📏 ШИРОКИЙ СПРЕД (низкая ликвидность)")
        
        # Анализ недельного тренда
        if weekly_trend['trend'] == 'восходящий':
            signals.append(f"📈 НЕДЕЛЬНЫЙ ТРЕНД ВВЕРХ (сила: {weekly_trend['strength']}%)")
        elif weekly_trend['trend'] == 'нисходящий':
            signals.append(f"📉 НЕДЕЛЬНЫЙ ТРЕНД ВНИЗ (сила: {weekly_trend['strength']}%)")
        
        # Анализ импульса
        if not hist_data.empty and len(hist_data) > 5:
            current_price = current_data['last']
            recent_high = hist_data['high'].tail(5).max()
            recent_low = hist_data['low'].tail(5).min()
            
            if current_price >= recent_high * 0.995:
                signals.append("🚀 ПРИБЛИЖЕНИЕ К МАКСИМУМАМ")
            elif current_price <= recent_low * 1.005:
                signals.append("📉 ПРИБЛИЖЕНИЕ К МИНИМУМАМ")
        
        return signals
    
    def generate_trading_ranges(self, symbol: str) -> Optional[Dict]:
        """Генерация торговых диапазонов для указанной акции"""
        print(f"🔄 Анализ {symbol}...")
        
        # Получаем данные
        current_data = self.get_stock_data(symbol)
        if not current_data:
            print(f"❌ Не удалось получить данные для {symbol}")
            return None
        
        hist_data = self.get_historical_data(symbol, days=30)
        if hist_data is None or hist_data.empty:
            print(f"❌ Не удалось получить исторические данные для {symbol}")
            return None
        
        orderbook = self.get_orderbook(symbol)
        
        # Расчет уровней
        tech_levels = self.calculate_technical_levels(hist_data)
        # Анализ недельного тренда
        weekly_trend = self.analyze_weekly_trend(hist_data)
        signals = self.analyze_intentionality(symbol, current_data, hist_data, orderbook, weekly_trend)
        
        current_price = current_data['last']
        
        # Определяем торговые диапазоны
        buy_zone_upper = tech_levels['support_1']
        sell_zone_lower = tech_levels['resistance_1']
        
        # Корректируем с учетом ATR
        atr = tech_levels.get('atr', current_price * 0.02)
        buy_zone_lower = max(0, buy_zone_upper - atr * 1.5)
        sell_zone_upper = sell_zone_lower + atr * 1.5
        
        # Генерация рекомендации
        if current_price <= buy_zone_upper:
            recommendation = "🟢 ПОКУПАТЬ"
            confidence = 0.7
        elif current_price >= sell_zone_lower:
            recommendation = "🔴 ПРОДАВАТЬ" 
            confidence = 0.7
        else:
            recommendation = "🟡 ДЕРЖАТЬ"
            confidence = 0.5
        
        # Увеличиваем уверенность при сильных сигналах и тренде
        strong_signals = [s for s in signals if '💥' in s or '🚀' in s or '📉' in s]
        if strong_signals:
            confidence = min(confidence + 0.2, 0.9)
        
        # Корректировка по тренду
        if weekly_trend['trend'] == 'восходящий' and recommendation == 'ПОКУПАТЬ':
            confidence = min(confidence + 0.1, 0.95)
        elif weekly_trend['trend'] == 'нисходящий' and recommendation == 'ПРОДАВАТЬ':
            confidence = min(confidence + 0.1, 0.95)
        
        return {
            'symbol': symbol,
            'name': self.available_stocks.get(symbol, {}).get('name', 'N/A'),
            'timestamp': datetime.now(),
            'current_price': current_price,
            'change': current_data.get('change', 0),
            'ranges': {
                'buy_zone': {
                    'lower': round(buy_zone_lower, 2),
                    'upper': round(buy_zone_upper, 2)
                },
                'sell_zone': {
                    'lower': round(sell_zone_lower, 2),
                    'upper': round(sell_zone_upper, 2)
                }
            },
            'technical_levels': tech_levels,
            'weekly_trend': weekly_trend,
            'signals': signals,
            'recommendation': recommendation,
            'confidence': confidence,
            'volume': current_data.get('volume', 0)
        }
    
    def analyze_multiple_stocks(self, symbols: List[str]) -> List[Dict]:
        """Анализ нескольких акций одновременно"""
        results = []
        
        for symbol in symbols:
            analysis = self.generate_trading_ranges(symbol)
            if analysis:
                results.append(analysis)
            time.sleep(0.5)  # Пауза между запросами
        
        # Сортируем по уверенности рекомендации
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def print_analysis(self, analysis: Optional[Dict]):
        """Красивый вывод анализа"""
        if not analysis:
            return
            
        print(f"\n{'='*70}")
        print(f"🎯 {analysis['symbol']} - {analysis['name']}")
        print(f"{'='*70}")
        
        change_str = f"+{analysis['change']:.2f}%" if analysis['change'] > 0 else f"{analysis['change']:.2f}%"
        print(f"💰 Цена: {analysis['current_price']} RUB ({change_str})")
        print(f"📊 Решение: {analysis['recommendation']} (уверенность: {analysis['confidence']:.0%})")
        
        print(f"\n📈 ТОРГОВЫЕ ДИАПАЗОНЫ:")
        ranges = analysis['ranges']
        print(f"   🟢 ПОКУПКА:  {ranges['buy_zone']['lower']} - {ranges['buy_zone']['upper']} RUB")
        print(f"   🟡 НЕЙТРАЛЬНО: {ranges['buy_zone']['upper']} - {ranges['sell_zone']['lower']} RUB")  
        print(f"   🔴 ПРОДАЖА: {ranges['sell_zone']['lower']} - {ranges['sell_zone']['upper']} RUB")
        
        print(f"\n🎯 КЛЮЧЕВЫЕ УРОВНИ:")
        levels = analysis['technical_levels']
        print(f"   Pivot: {levels['pivot']} | R1: {levels['resistance_1']} | S1: {levels['support_1']}")
        if levels['supports']:
            print(f"   Поддержки: {levels['supports']}")
        if levels['resistances']:
            print(f"   Сопротивления: {levels['resistances']}")
        print(f"   ATR (волатильность): {levels['atr']}")
        
        trend = analysis.get('weekly_trend', {})
        if trend:
            print(f"   Недельный тренд: {trend.get('trend', 'N/A')} (сила: {trend.get('strength', 0)}%)")
        
        if analysis['signals']:
            print(f"\n📡 СИГНАЛЫ:")
            for signal in analysis['signals']:
                print(f"   • {signal}")
        
        print(f"\n📊 Объем: {analysis['volume']:,.0f}")
        print(f"🕒 Время анализа: {analysis['timestamp'].strftime('%H:%M:%S')}")
        print(f"{'='*70}")

# Управляющий класс для взаимодействия с пользователем
class TradingBotInterface:
    def __init__(self):
        self.bot = MoexTradingBot()
        self.favorite_stocks = []
    
    def show_main_menu(self):
        """Главное меню"""
        while True:
            print(f"\n{'='*50}")
            print("🤖 УНИВЕРСАЛЬНЫЙ ТРЕЙДИНГ БОТ MOEX")
            print(f"{'='*50}")
            print("1 - Анализ одной акции")
            print("2 - Анализ нескольких акций")
            print("3 - Поиск акций")
            print("4 - Мои избранные акции")
            print("5 - Топ-10 рекомендаций")
            print("6 - Выход")
            
            choice = input("\nВыберите действие: ").strip()
            
            if choice == "1":
                self.analyze_single_stock()
            elif choice == "2":
                self.analyze_multiple_stocks()
            elif choice == "3":
                self.search_stocks()
            elif choice == "4":
                self.manage_favorites()
            elif choice == "5":
                self.top_recommendations()
            elif choice == "6":
                print("👋 До свидания!")
                break
            else:
                print("❌ Неверный выбор")
    
    def analyze_single_stock(self):
        """Анализ одной акции"""
        symbol = input("Введите тикер акции (например: SBER, GAZP, YNDX): ").strip().upper()
        
        if symbol not in self.bot.available_stocks:
            print(f"❌ Акция {symbol} не найдена или нет данных")
            return
        
        analysis = self.bot.generate_trading_ranges(symbol)
        self.bot.print_analysis(analysis)
        
        # Предлагаем добавить в избранное
        if analysis and input("Добавить в избранные? (y/n): ").lower() == 'y':
            if symbol not in self.favorite_stocks:
                self.favorite_stocks.append(symbol)
                print(f"✅ {symbol} добавлен в избранные")
    
    def analyze_multiple_stocks(self):
        """Анализ нескольких акций"""
        if not self.favorite_stocks:
            print("❌ Сначала добавьте акции в избранные")
            return
        
        print("\n📊 Анализ избранных акций:")
        for i, symbol in enumerate(self.favorite_stocks, 1):
            print(f"  {i}. {symbol} - {self.bot.available_stocks[symbol]['name']}")
        
        symbols_input = input("Введите номера акций через пробел (или Enter для всех): ").strip()
        
        if symbols_input:
            try:
                indices = [int(x)-1 for x in symbols_input.split()]
                symbols = [self.favorite_stocks[i] for i in indices if i < len(self.favorite_stocks)]
            except:
                print("❌ Ошибка ввода")
                return
        else:
            symbols = self.favorite_stocks
        
        print(f"\n🔍 Анализ {len(symbols)} акций...")
        results = self.bot.analyze_multiple_stocks(symbols)
        
        print(f"\n📋 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        for analysis in results:
            self.bot.print_analysis(analysis)
    
    def search_stocks(self):
        """Поиск акций"""
        query = input("Введите название или тикер для поиска: ").strip()
        
        if not query:
            return
        
        results = self.bot.search_stocks(query)
        
        if not results:
            print("❌ Ничего не найдено")
            return
        
        print(f"\n🔍 Найдено акций: {len(results)}")
        for i, (ticker, info) in enumerate(results.items(), 1):
            print(f"  {i}. {ticker} - {info['name']} - {info['price']} RUB")
        
        if input("\nДобавить найденные акции в избранные? (y/n): ").lower() == 'y':
            for ticker in results.keys():
                if ticker not in self.favorite_stocks:
                    self.favorite_stocks.append(ticker)
            print(f"✅ Добавлено {len(results)} акций в избранные")
    
    def manage_favorites(self):
        """Управление избранными акциями"""
        if not self.favorite_stocks:
            print("📝 Список избранных акций пуст")
            return
        
        print(f"\n⭐ МОИ ИЗБРАННЫЕ АКЦИИ ({len(self.favorite_stocks)}):")
        for i, symbol in enumerate(self.favorite_stocks, 1):
            info = self.bot.available_stocks.get(symbol, {})
            print(f"  {i}. {symbol} - {info.get('name', 'N/A')}")
        
        print("\n1 - Удалить акцию")
        print("2 - Очистить список")
        print("3 - Проанализировать все")
        
        choice = input("Выберите действие: ").strip()
        
        if choice == "1":
            try:
                idx = int(input("Номер акции для удаления: ")) - 1
                if 0 <= idx < len(self.favorite_stocks):
                    removed = self.favorite_stocks.pop(idx)
                    print(f"✅ {removed} удален из избранных")
            except:
                print("❌ Ошибка ввода")
        elif choice == "2":
            self.favorite_stocks.clear()
            print("✅ Список избранных очищен")
        elif choice == "3":
            self.analyze_multiple_stocks()
    
    def top_recommendations(self):
        """Топ-10 рекомендаций из голубых фишек"""
        blue_chips = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'NVTK', 'TATN', 'GMKN', 'PLZL', 'ALRS', 'MGNT']
        
        print(f"\n🏆 АНАЛИЗ ГОЛУБЫХ ФИШЕК ({len(blue_chips)} акций)")
        results = self.bot.analyze_multiple_stocks(blue_chips)
        
        # Группируем по рекомендациям
        buy_recommendations = [r for r in results if 'ПОКУПАТЬ' in r['recommendation']]
        sell_recommendations = [r for r in results if 'ПРОДАВАТЬ' in r['recommendation']]
        hold_recommendations = [r for r in results if 'ДЕРЖАТЬ' in r['recommendation']]
        
        print(f"\n🟢 ПОКУПАТЬ ({len(buy_recommendations)}):")
        for analysis in sorted(buy_recommendations, key=lambda x: x['confidence'], reverse=True):
            print(f"   {analysis['symbol']} - уверенность {analysis['confidence']:.0%}")
        
        print(f"\n🔴 ПРОДАВАТЬ ({len(sell_recommendations)}):")
        for analysis in sorted(sell_recommendations, key=lambda x: x['confidence'], reverse=True):
            print(f"   {analysis['symbol']} - уверенность {analysis['confidence']:.0%}")
        
        print(f"\n🟡 ДЕРЖАТЬ ({len(hold_recommendations)}):")
        for analysis in hold_recommendations:
            print(f"   {analysis['symbol']}")

# Запуск бота
if __name__ == "__main__":
    print("🚀 Загрузка универсального трейдинг бота...")
    
    interface = TradingBotInterface()
    interface.show_main_menu()