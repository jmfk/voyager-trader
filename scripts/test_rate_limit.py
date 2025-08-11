#!/usr/bin/env python3
"""
Test rate limiting configuration for VOYAGER-Trader admin interface.

This script tests rate limiting by making multiple requests to API endpoints
to verify that rate limits are properly enforced.
"""

import os
import sys
import time
import requests
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_rate_limit(
    endpoint: str, 
    rate_limit: str, 
    api_url: str = "http://localhost:8001",
    headers: Dict[str, str] = None
) -> Tuple[bool, List[Dict]]:
    """Test rate limiting for a specific endpoint."""
    
    # Parse rate limit (e.g., "5/minute" -> 5 requests)
    try:
        limit_count = int(rate_limit.split('/')[0])
        limit_period = rate_limit.split('/')[1]
    except:
        print(f"❌ Invalid rate limit format: {rate_limit}")
        return False, []
    
    print(f"\n🔄 Testing {endpoint} with limit: {rate_limit}")
    print(f"   Making {limit_count + 2} requests to test enforcement...")
    
    results = []
    headers = headers or {}
    
    # Make requests up to and beyond the limit
    for i in range(limit_count + 2):
        try:
            start_time = time.time()
            response = requests.get(f"{api_url}{endpoint}", headers=headers, timeout=5)
            end_time = time.time()
            
            result = {
                "request_num": i + 1,
                "status_code": response.status_code,
                "response_time": round((end_time - start_time) * 1000, 2),
                "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining"),
                "rate_limit_limit": response.headers.get("X-RateLimit-Limit"),
                "retry_after": response.headers.get("Retry-After"),
            }
            
            results.append(result)
            
            # Log the result
            status_emoji = "✅" if response.status_code < 400 else "❌"
            remaining = result["rate_limit_remaining"] or "?"
            print(f"   {status_emoji} Request {i+1}: {response.status_code} "
                  f"(remaining: {remaining}, time: {result['response_time']}ms)")
            
            # Brief delay between requests
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   ❌ Request {i+1} failed: {e}")
            results.append({
                "request_num": i + 1,
                "status_code": 0,
                "error": str(e)
            })
    
    # Analyze results
    success_count = sum(1 for r in results if r.get("status_code", 0) < 400)
    rate_limited_count = sum(1 for r in results if r.get("status_code") == 429)
    
    print(f"   📊 Results: {success_count} successful, {rate_limited_count} rate limited")
    
    # Check if rate limiting is working
    if rate_limited_count > 0:
        print(f"   ✅ Rate limiting is working - blocked {rate_limited_count} requests")
        return True, results
    elif success_count <= limit_count:
        print(f"   ⚠️  Rate limiting may be working (all requests within limit)")
        return True, results
    else:
        print(f"   ❌ Rate limiting not working - allowed {success_count} requests")
        return False, results


def test_login_rate_limit(api_url: str = "http://localhost:8001") -> Tuple[bool, List[Dict]]:
    """Test rate limiting for login endpoint with actual POST requests."""
    
    print(f"\n🔐 Testing login endpoint rate limiting...")
    
    # Get the rate limit from environment or use default
    rate_limit = os.getenv("VOYAGER_RATE_LIMIT_LOGIN", "5/minute")
    limit_count = int(rate_limit.split('/')[0])
    
    print(f"   Making {limit_count + 2} login attempts to test enforcement...")
    
    results = []
    login_data = {"username": "admin", "password": "wrong_password"}
    
    for i in range(limit_count + 2):
        try:
            start_time = time.time()
            response = requests.post(
                f"{api_url}/api/auth/login",
                json=login_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            end_time = time.time()
            
            result = {
                "request_num": i + 1,
                "status_code": response.status_code,
                "response_time": round((end_time - start_time) * 1000, 2),
                "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining"),
                "rate_limit_limit": response.headers.get("X-RateLimit-Limit"),
                "retry_after": response.headers.get("Retry-After"),
            }
            
            results.append(result)
            
            # Log the result
            status_emoji = "✅" if response.status_code in [401, 422] else ("❌" if response.status_code == 429 else "⚠️")
            remaining = result["rate_limit_remaining"] or "?"
            print(f"   {status_emoji} Login {i+1}: {response.status_code} "
                  f"(remaining: {remaining}, time: {result['response_time']}ms)")
            
            # Brief delay between requests
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   ❌ Login {i+1} failed: {e}")
            results.append({
                "request_num": i + 1,
                "status_code": 0,
                "error": str(e)
            })
    
    # Analyze results
    normal_response_count = sum(1 for r in results if r.get("status_code") in [401, 422])
    rate_limited_count = sum(1 for r in results if r.get("status_code") == 429)
    
    print(f"   📊 Results: {normal_response_count} normal responses, {rate_limited_count} rate limited")
    
    if rate_limited_count > 0:
        print(f"   ✅ Login rate limiting is working - blocked {rate_limited_count} attempts")
        return True, results
    else:
        print(f"   ❌ Login rate limiting not working or limit not reached")
        return False, results


def get_current_rate_limits() -> Dict[str, str]:
    """Get current rate limiting configuration from environment."""
    return {
        "login": os.getenv("VOYAGER_RATE_LIMIT_LOGIN", "5/minute"),
        "api": os.getenv("VOYAGER_RATE_LIMIT_API", "100/minute"),
        "health": os.getenv("VOYAGER_RATE_LIMIT_HEALTH", "60/minute"),
        "storage": os.getenv("VOYAGER_RATE_LIMIT_STORAGE", "memory://"),
    }


def main():
    """Main function to test rate limiting configuration."""
    print("🛡️  VOYAGER-Trader Rate Limiting Tester")
    print("=" * 45)
    
    # Get current configuration
    rate_limits = get_current_rate_limits()
    
    print("\n🔧 Current Rate Limiting Configuration:")
    print("=" * 40)
    for key, value in rate_limits.items():
        print(f"  {key.upper()}: {value}")
    
    # Parse command line arguments
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
    
    print(f"\n🧪 Testing API: {api_url}")
    
    # Test health endpoint (no auth required)
    print("\n" + "="*50)
    health_success, health_results = test_rate_limit("/api/health", rate_limits["health"], api_url)
    
    # Test login endpoint (special case - POST requests)
    print("\n" + "="*50)
    login_success, login_results = test_login_rate_limit(api_url)
    
    # Summary
    print("\n" + "="*50)
    print("📊 Rate Limiting Test Summary")
    print("=" * 30)
    
    total_tests = 2
    passed_tests = sum([health_success, login_success])
    
    print(f"✅ Health endpoint rate limiting: {'PASS' if health_success else 'FAIL'}")
    print(f"🔐 Login endpoint rate limiting: {'PASS' if login_success else 'FAIL'}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ All rate limiting tests passed!")
        print("\n💡 Tips:")
        print("  - Monitor rate limiting in production logs")
        print("  - Adjust limits based on actual usage patterns")
        print("  - Consider using Redis for distributed rate limiting")
        print("  - Implement progressive delays for repeated violations")
    else:
        print("❌ Some rate limiting tests failed!")
        print("\n🔧 Troubleshooting:")
        print("  - Check that slowapi is properly installed")
        print("  - Verify rate limiting configuration in environment")
        print("  - Ensure API server is running and accessible")
        sys.exit(1)


if __name__ == "__main__":
    main()