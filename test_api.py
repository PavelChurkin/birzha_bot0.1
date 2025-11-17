import requests
import json

# Test securities API
url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
params = {
    'iss.only': 'securities',
    'securities.columns': 'SECID,SHORTNAME,SECNAME,PREVPRICE'
}

try:
    response = requests.get(url, params=params)
    data = response.json()
    print("Securities API works")
    print("Columns:", data['securities']['columns'])
    print("Sample data:", data['securities']['data'][:2])
except Exception as e:
    print(f"Error: {e}")

# Test marketdata API for SBER
url = "https://iss.moex.com/iss/engines/stock/markets/shares/securities/SBER.json"
params = {
    'iss.only': 'marketdata',
    'marketdata.columns': 'BOARDID,LAST,OPEN,HIGH,LOW,VOLTODAY,VALTODAY,LASTTOPREVPRICE'
}

try:
    response = requests.get(url, params=params)
    data = response.json()
    print("\nMarketdata API works")
    print("Columns:", data['marketdata']['columns'])
    print("Sample data:", data['marketdata']['data'][:1])
except Exception as e:
    print(f"Error: {e}")