#!/usr/bin/env python3
"""
Test script to verify the admin interface setup.
"""

import sys
from pathlib import Path

def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        "src/voyager_trader/admin_api.py",
        "admin-ui/package.json",
        "admin-ui/src/App.js",
        "admin-ui/src/index.js",
        "admin-ui/src/services/api.js",
        "admin-ui/src/components/Dashboard.js",
        "admin-ui/src/components/Login.js",
        "start_admin.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print("\n✗ Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✓ All required files present")
    return True


def test_python_imports():
    """Test that Python modules can be imported."""
    print("\nTesting Python imports...")
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        # Test core modules
        from voyager_trader.core import VoyagerTrader
        print("✓ VoyagerTrader core imports")
        
        # Test if FastAPI is available
        try:
            import fastapi
            import uvicorn
            print("✓ FastAPI dependencies available")
            
            # Test admin API import
            from voyager_trader.admin_api import app
            print("✓ Admin API imports successfully")
            
        except ImportError as e:
            print(f"✗ FastAPI not installed: {e}")
            print("  Run: pip install -r requirements.txt")
            return False
            
    except ImportError as e:
        print(f"✗ Core import error: {e}")
        return False
    
    return True


def test_react_setup():
    """Test React app setup."""
    print("\nTesting React setup...")
    
    admin_ui_path = Path("admin-ui")
    package_json = admin_ui_path / "package.json"
    
    if not package_json.exists():
        print("✗ package.json not found")
        return False
    
    print("✓ package.json found")
    
    # Check if node_modules exists
    node_modules = admin_ui_path / "node_modules"
    if node_modules.exists():
        print("✓ node_modules directory exists")
    else:
        print("! node_modules not found - run 'npm install' in admin-ui/")
    
    return True


def main():
    """Run all tests."""
    print("VoyagerTrader Admin Interface Setup Test")
    print("=" * 50)
    
    success = True
    
    success &= test_file_structure()
    success &= test_python_imports()
    success &= test_react_setup()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ Setup test completed successfully!")
        print("\nNext steps:")
        print("1. Install Python dependencies: pip install -r requirements.txt")
        print("2. Install React dependencies: cd admin-ui && npm install")
        print("3. Start the admin interface: python start_admin.py")
    else:
        print("✗ Setup test failed - see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()