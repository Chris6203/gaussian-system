#!/usr/bin/env python3
"""Quick FMP API test - run directly to test intraday endpoint"""
import requests

api_key = 'tkkQOoWrs7lAgB8Qjm05jEZz6QrrfLMv'

# Test the CORRECT stable API endpoint format
url = f'https://financialmodelingprep.com/stable/historical-chart/1min?symbol=AAPL&apikey={api_key}'

print("Testing FMP Historical Intraday Endpoint")
print("="*50)
print(f"URL: {url[:80]}...")

try:
    r = requests.get(url, timeout=15)
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list):
            print(f"[SUCCESS] Got {len(data)} records!")
            if len(data) > 0:
                print(f"First record: {data[0]}")
        else:
            print(f"Response: {str(data)[:300]}")
    else:
        print(f"Error: {r.text[:300]}")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50)
print("If this works, the fetch_6months.py script should work too!")



