#!/usr/bin/env python3
"""Quick test of FMP API"""
import requests
import json

api_key = "tkkQOoWrs7lAgB8Qjm05jEZz6QrrfLMv"

# Test 1: Quote endpoint (stable API)
print("Test 1: Quote endpoint")
url = f"https://financialmodelingprep.com/stable/quote?symbol=SPY&apikey={api_key}"
print(f"URL: {url}")
try:
    resp = requests.get(url, timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60 + "\n")

# Test 2: Historical intraday endpoint (v3 API)
print("Test 2: Historical intraday (1min)")
url = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/SPY?apikey={api_key}"
print(f"URL: {url}")
try:
    resp = requests.get(url, timeout=10)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    if isinstance(data, list):
        print(f"Records: {len(data)}")
        if data:
            print(f"First record: {data[0]}")
            print(f"Last record: {data[-1]}")
    else:
        print(f"Response: {data}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60 + "\n")

# Test 3: Check if there's a different endpoint for longer history
print("Test 3: Historical daily (full)")
url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=SPY&apikey={api_key}"
print(f"URL: {url}")
try:
    resp = requests.get(url, timeout=10)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    if 'historical' in data:
        print(f"Records: {len(data['historical'])}")
        if data['historical']:
            print(f"First: {data['historical'][0]}")
    else:
        print(f"Response: {str(data)[:500]}")
except Exception as e:
    print(f"Error: {e}")



