#!/usr/bin/env python3
"""Comprehensive API endpoint testing script."""

import httpx
import asyncio
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/api"


async def test_endpoints():
    """Test critical API endpoints and analyze results."""
    results = {"passed": [], "failed": [], "warnings": []}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Health endpoints
        endpoints_to_test = [
            ("GET", "/health", "Health check"),
            ("GET", "/health/db", "Database health"),
            ("GET", "/health/cache", "Cache health"),
            ("GET", "/symbols", "Symbols list"),
            ("GET", "/dips/ranking", "Dip ranking"),
            ("GET", "/dips/states", "Dip states"),
            ("GET", "/dips/benchmarks", "Benchmarks"),
            ("GET", "/cronjobs", "Cronjobs list"),
            ("GET", "/calendar/summary", "Calendar summary"),
            ("GET", "/market/sectors", "Market sectors"),
            ("GET", "/dipfinder/signals", "Dipfinder signals"),
            ("GET", "/dipfinder/config", "Dipfinder config"),
            ("GET", "/quant/recommendations", "Quant recommendations"),
            ("GET", "/signals/strategy", "Strategy signals"),
            ("GET", "/notifications/trigger-types", "Notification triggers"),
            ("GET", "/ai-personas", "AI Personas"),
        ]
        
        print("=" * 70)
        print("API ENDPOINT TESTS")
        print("=" * 70)
        
        for method, path, name in endpoints_to_test:
            try:
                if method == "GET":
                    resp = await client.get(f"{BASE_URL}{path}")
                elif method == "POST":
                    resp = await client.post(f"{BASE_URL}{path}")
                
                status = "✅" if resp.status_code < 400 else "❌"
                
                # Check response content
                try:
                    data = resp.json()
                    if isinstance(data, list):
                        count = len(data)
                        detail = f"{count} items"
                    elif isinstance(data, dict):
                        if "data" in data and isinstance(data["data"], list):
                            count = len(data["data"])
                            detail = f"{count} items"
                        else:
                            detail = f"{len(data)} keys"
                    else:
                        detail = str(data)[:50]
                except Exception:
                    detail = f"{len(resp.content)} bytes"
                
                print(f"{status} {name:35} [{resp.status_code}] {detail}")
                
                if resp.status_code < 400:
                    results["passed"].append((path, resp.status_code))
                else:
                    results["failed"].append((path, resp.status_code, resp.text[:100]))
                    
            except Exception as e:
                print(f"❌ {name:35} [ERROR] {str(e)[:50]}")
                results["failed"].append((path, "ERROR", str(e)))
        
        # Test specific symbol endpoints
        print("\n" + "=" * 70)
        print("SYMBOL-SPECIFIC ENDPOINT TESTS (SPY, MSFT, NFLX)")
        print("=" * 70)
        
        test_symbols = ["SPY", "MSFT", "NFLX"]
        symbol_endpoints = [
            "/symbols/{symbol}",
            "/symbols/fundamentals/{symbol}",
            "/dips/{symbol}/info",
            "/dips/{symbol}/state",
            "/dipfinder/signals/{symbol}",
            "/signals/strategy/{symbol}",
            "/quant/recommendations/{symbol}/dip-analysis",
            "/quant/recommendations/{symbol}/backtest-v2",
        ]
        
        for symbol in test_symbols:
            print(f"\n--- {symbol} ---")
            for endpoint_template in symbol_endpoints:
                path = endpoint_template.replace("{symbol}", symbol)
                name = endpoint_template.split("/")[-1].replace("{symbol}", symbol)
                try:
                    resp = await client.get(f"{BASE_URL}{path}")
                    status = "✅" if resp.status_code < 400 else "⚠️" if resp.status_code == 404 else "❌"
                    
                    try:
                        data = resp.json()
                        if isinstance(data, dict) and "error" in data:
                            detail = f"Error: {data['error'][:40]}..."
                        elif isinstance(data, dict):
                            detail = f"{len(data)} keys"
                        else:
                            detail = str(data)[:40]
                    except Exception:
                        detail = f"{resp.status_code}"
                    
                    print(f"  {status} {name:35} {detail}")
                    
                except Exception as e:
                    print(f"  ❌ {name:35} {str(e)[:30]}")
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Passed: {len(results['passed'])}")
        print(f"Failed: {len(results['failed'])}")
        if results["failed"]:
            print("\nFailed endpoints:")
            for item in results["failed"]:
                print(f"  - {item[0]}: {item[1]}")


if __name__ == "__main__":
    asyncio.run(test_endpoints())
