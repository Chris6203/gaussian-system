#!/usr/bin/env python3
"""Quick test of Polygon API key"""
import requests
import json

# Load API key
with open('config.json', 'r') as f:
    config = json.load(f)
api_key = config['credentials']['polygon']['api_key']

print("="*60)
print("POLYGON API KEY TEST")
print("="*60)
print(f"API Key: {api_key}")
print()

# Test 1: Simple stock quote
print("TEST 1: Stock quote (AAPL)")
url = f"https://api.polygon.io/v2/last/trade/AAPL?apiKey={api_key}"
try:
    resp = requests.get(url, timeout=10)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        print("  [OK] Quote endpoint works!")
    else:
        print(f"  [FAIL] {resp.text[:300]}")
except Exception as e:
    print(f"  [ERROR] {e}")
print()

# Test 2: Historical aggregates (recent data)
print("TEST 2: Historical 1-min data (last week)")
url2 = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/minute/2024-12-01/2024-12-06?apiKey={api_key}&limit=10"
try:
    resp2 = requests.get(url2, timeout=10)
    print(f"  Status: {resp2.status_code}")
    if resp2.status_code == 200:
        data = resp2.json()
        count = data.get('resultsCount', 0)
        print(f"  [OK] Got {count} results!")
        if data.get('results'):
            print(f"  Sample: {data['results'][0]}")
    else:
        print(f"  [FAIL] {resp2.text[:300]}")
except Exception as e:
    print(f"  [ERROR] {e}")
print()

# Test 3: Check API status
print("TEST 3: API Status")
url3 = f"https://api.polygon.io/v1/marketstatus/now?apiKey={api_key}"
try:
    resp3 = requests.get(url3, timeout=10)
    print(f"  Status: {resp3.status_code}")
    if resp3.status_code == 200:
        print("  [OK] API status works!")
    else:
        print(f"  [FAIL] {resp3.text[:300]}")
except Exception as e:
    print(f"  [ERROR] {e}")

print()
print("="*60)
if resp.status_code == 200 and resp2.status_code == 200:
    print("ALL TESTS PASSED - Ready to fetch historical data!")
else:
    print("Some tests failed - check API key and plan status")
print("="*60)



