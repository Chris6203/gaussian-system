#!/usr/bin/env python3
"""
Test FMP API key and endpoints - Testing NEW STABLE API
FMP deprecated v3 API on Aug 31, 2025 - must use /stable/ endpoints
"""

import requests
import json

# Load API key from config
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    api_key = config.get('credentials', {}).get('fmp', {}).get('api_key')
    print(f"API Key from config: {api_key}" if api_key else "No API key found!")
except Exception as e:
    print(f"Error loading config: {e}")
    api_key = None

if not api_key:
    print("\nPlease add your FMP API key to config.json under credentials.fmp.api_key")
    exit(1)

print("\n" + "="*60)
print("Testing FMP STABLE API Endpoints (v3 is deprecated)")
print("="*60)

def test_endpoint(name, url):
    print(f"\n[{name}]")
    print(f"URL: {url}")
    try:
        resp = requests.get(url, timeout=15)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                if len(data) > 0:
                    print(f"[OK] Got {len(data)} records")
                    print(f"First record: {str(data[0])[:200]}")
                    return True, data
                else:
                    print("[FAIL] Empty list returned")
            elif isinstance(data, dict):
                if 'Error Message' in str(data):
                    print(f"[FAIL] Error: {str(data)[:200]}")
                elif data:
                    print(f"[OK] Response: {str(data)[:200]}")
                    return True, data
                else:
                    print("[FAIL] Empty dict")
            else:
                print(f"Response: {str(data)[:150]}")
        else:
            print(f"[FAIL] {resp.text[:300]}")
        return False, None
    except Exception as e:
        print(f"[ERROR] {e}")
        return False, None

# Test different STABLE API endpoints
results = {}

# Test 1: Quote (stable) - WORKING
results['quote'] = test_endpoint(
    "Quote (stable API)", 
    f"https://financialmodelingprep.com/stable/quote?symbol=AAPL&apikey={api_key}"
)[0]

# Test 2: Historical EOD (stable)
results['historical_eod'] = test_endpoint(
    "Historical EOD (stable)",
    f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=AAPL&apikey={api_key}"
)[0]

# Test 3: Historical Intraday - try different formats
print("\n" + "-"*60)
print("Testing intraday historical endpoints...")
print("-"*60)

# Format A: CORRECT FORMAT - /historical-chart/1min?symbol=XXX (per FMP docs)
results['intraday_A'] = test_endpoint(
    "Intraday A: /historical-chart/1min?symbol=XXX (CORRECT)",
    f"https://financialmodelingprep.com/stable/historical-chart/1min?symbol=AAPL&apikey={api_key}"
)[0]

# Format B: Try 5min interval
results['intraday_B'] = test_endpoint(
    "Intraday B: /historical-chart/5min?symbol=XXX",
    f"https://financialmodelingprep.com/stable/historical-chart/5min?symbol=AAPL&apikey={api_key}"
)[0]

# Format C: Try with date params
results['intraday_C'] = test_endpoint(
    "Intraday C: /historical-chart/1min with date range",
    f"https://financialmodelingprep.com/stable/historical-chart/1min?symbol=SPY&from=2024-12-01&to=2024-12-06&apikey={api_key}"
)[0]

# Format D: Alternative endpoint test
results['intraday_D'] = test_endpoint(
    "Intraday D: /historical-price-full/1min",
    f"https://financialmodelingprep.com/stable/historical-price-full/1min?symbol=AAPL&apikey={api_key}"
)[0]

# Test 4: Stock list (stable)
results['stock_list'] = test_endpoint(
    "Stock list (stable)",
    f"https://financialmodelingprep.com/stable/stock-list?apikey={api_key}"
)[0]

# Test 5: Profile (stable)
results['profile'] = test_endpoint(
    "Profile (stable)",
    f"https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey={api_key}"
)[0]

# Test 6: Search
results['search'] = test_endpoint(
    "Search symbol (stable)",
    f"https://financialmodelingprep.com/stable/search-symbol?query=AAPL&apikey={api_key}"
)[0]

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for name, success in results.items():
    status = "[OK]" if success else "[FAIL]"
    print(f"  {status} {name}")

intraday_works = any(results.get(k) for k in ['intraday_A', 'intraday_B', 'intraday_C', 'intraday_D'])
print("\n" + "="*60)
if intraday_works:
    print("INTRADAY DATA IS AVAILABLE!")
    print("We can fetch 1-minute historical data.")
else:
    print("Intraday data NOT available with current endpoints.")
    print("The Starter plan may not include 1-min historical data,")
    print("or FMP uses a different endpoint format.")
print("="*60)



