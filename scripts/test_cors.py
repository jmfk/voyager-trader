#!/usr/bin/env python3
"""
Test CORS configuration for VOYAGER-Trader admin interface.

This script tests CORS configuration by making requests from different origins
to verify that the admin API properly handles CORS policies.
"""

import os
import sys
import requests
from typing import List


def test_cors_configuration(api_url: str = "http://localhost:8001", origins: List[str] = None):
    """Test CORS configuration with different origins."""
    
    if origins is None:
        origins = [
            "http://localhost:3001",
            "http://127.0.0.1:3001", 
            "https://admin.example.com",
            "https://malicious.com",
        ]
    
    print(f"🔍 Testing CORS configuration for API: {api_url}")
    print("=" * 60)
    
    # Test health endpoint (should work without CORS for public access)
    try:
        response = requests.get(f"{api_url}/api/health", timeout=5)
        print(f"✅ Health endpoint accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False
    
    # Test CORS preflight for each origin
    for origin in origins:
        print(f"\n🌐 Testing origin: {origin}")
        
        # CORS preflight request
        headers = {
            'Origin': origin,
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type',
        }
        
        try:
            response = requests.options(
                f"{api_url}/api/auth/login",
                headers=headers,
                timeout=5
            )
            
            cors_headers = {
                'access-control-allow-origin': response.headers.get('access-control-allow-origin'),
                'access-control-allow-methods': response.headers.get('access-control-allow-methods'),
                'access-control-allow-headers': response.headers.get('access-control-allow-headers'),
                'access-control-allow-credentials': response.headers.get('access-control-allow-credentials'),
            }
            
            if response.status_code == 200 and cors_headers['access-control-allow-origin']:
                print(f"  ✅ CORS allowed - Status: {response.status_code}")
                print(f"     Allow-Origin: {cors_headers['access-control-allow-origin']}")
                print(f"     Allow-Methods: {cors_headers['access-control-allow-methods']}")
                print(f"     Allow-Credentials: {cors_headers['access-control-allow-credentials']}")
            else:
                print(f"  ❌ CORS blocked - Status: {response.status_code}")
                print(f"     Allow-Origin: {cors_headers['access-control-allow-origin']}")
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
    
    return True


def get_current_cors_config():
    """Get current CORS configuration from environment."""
    cors_origins = os.getenv("VOYAGER_CORS_ORIGINS")
    
    print("\n🔧 Current CORS Configuration:")
    print("=" * 40)
    
    if cors_origins:
        origins = [origin.strip() for origin in cors_origins.split(",")]
        print(f"Environment variable: VOYAGER_CORS_ORIGINS")
        print(f"Configured origins:")
        for origin in origins:
            print(f"  - {origin}")
    else:
        print("No VOYAGER_CORS_ORIGINS environment variable set")
        print("Using default origins:")
        print("  - http://localhost:3001")
        print("  - http://127.0.0.1:3001")
    
    return cors_origins


def main():
    """Main function to test CORS configuration."""
    print("🔐 VOYAGER-Trader CORS Configuration Tester")
    print("=" * 50)
    
    # Get current configuration
    cors_config = get_current_cors_config()
    
    # Parse command line arguments
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
    
    # Custom origins for testing
    test_origins = None
    if cors_config:
        test_origins = [origin.strip() for origin in cors_config.split(",")]
        # Add some test origins for security verification
        test_origins.extend([
            "https://malicious.com",
            "http://evil.example.com"
        ])
    
    print(f"\n🧪 Running CORS Tests...")
    success = test_cors_configuration(api_url, test_origins)
    
    if success:
        print(f"\n✅ CORS testing completed successfully!")
        print("\n💡 Tips:")
        print("  - Allowed origins should only include trusted domains")
        print("  - Use HTTPS origins in production")
        print("  - Regularly audit CORS configuration")
        print(f"  - Set VOYAGER_CORS_ORIGINS for production deployment")
    else:
        print(f"\n❌ CORS testing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()