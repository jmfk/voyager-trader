#!/usr/bin/env python3
"""
Startup script for the VoyagerTrader Admin Interface.

This script starts both the FastAPI backend server and provides instructions
for starting the React frontend development server.
"""

import os
import sys
import subprocess
import signal
from pathlib import Path


def check_requirements():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        print("✓ FastAPI dependencies found")
    except ImportError:
        print("✗ FastAPI dependencies not found. Please install requirements:")
        print("  pip install -r requirements.txt")
        return False
    
    # Check if Node.js is available
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Node.js found: {result.stdout.strip()}")
        else:
            print("✗ Node.js not found. Please install Node.js to run the frontend.")
            return False
    except FileNotFoundError:
        print("✗ Node.js not found. Please install Node.js to run the frontend.")
        return False
    
    return True


def install_frontend_dependencies():
    """Install React app dependencies if needed."""
    admin_ui_path = Path(__file__).parent / "admin-ui"
    node_modules_path = admin_ui_path / "node_modules"
    
    if not node_modules_path.exists():
        print("Installing React app dependencies...")
        try:
            subprocess.run(['npm', 'install'], 
                         cwd=admin_ui_path, 
                         check=True)
            print("✓ React dependencies installed")
        except subprocess.CalledProcessError:
            print("✗ Failed to install React dependencies")
            return False
    else:
        print("✓ React dependencies already installed")
    
    return True


def start_backend():
    """Start the FastAPI backend server."""
    print("Starting VoyagerTrader Admin API server...")
    
    # Add the src directory to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        # Import and run the FastAPI app
        from voyager_trader.admin_api import app
        import uvicorn
        
        print("✓ Backend server starting on http://localhost:8001")
        print("  - API documentation: http://localhost:8001/docs")
        print("  - Health check: http://localhost:8001/api/health")
        
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
        
    except ImportError as e:
        print(f"✗ Failed to import admin API: {e}")
        print("Make sure you're in the correct directory and dependencies are installed.")
        return False
    except Exception as e:
        print(f"✗ Failed to start backend server: {e}")
        return False


def print_frontend_instructions():
    """Print instructions for starting the React frontend."""
    print("\n" + "="*60)
    print("TO START THE REACT FRONTEND:")
    print("="*60)
    print("Open a new terminal and run:")
    print(f"  cd {Path(__file__).parent}/admin-ui")
    print("  npm start")
    print()
    print("The React app will start on http://localhost:3001")
    print()
    print("DEFAULT LOGIN CREDENTIALS:")
    print("  Username: admin")
    print("  Password: admin123")
    print("="*60)


def main():
    """Main startup function."""
    print("VoyagerTrader Admin Interface Startup")
    print("="*40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Install frontend dependencies
    if not install_frontend_dependencies():
        sys.exit(1)
    
    # Print frontend instructions
    print_frontend_instructions()
    
    # Start backend server
    try:
        start_backend()
    except KeyboardInterrupt:
        print("\n✓ Admin interface shutdown complete")
        sys.exit(0)


if __name__ == "__main__":
    main()