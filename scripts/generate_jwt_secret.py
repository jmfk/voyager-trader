#!/usr/bin/env python3
"""
Generate a secure JWT secret for VOYAGER-Trader admin interface.

This script generates a cryptographically secure JWT secret and provides
options for persisting it to environment configuration.
"""

import os
import secrets
import sys
from pathlib import Path


def generate_jwt_secret() -> str:
    """Generate a cryptographically secure JWT secret."""
    return secrets.token_urlsafe(32)


def update_env_file(secret: str, env_file: Path = Path(".env")) -> bool:
    """Update .env file with JWT secret."""
    try:
        if env_file.exists():
            # Read existing content
            with open(env_file, 'r') as f:
                lines = f.readlines()
            
            # Check if JWT secret already exists
            jwt_secret_exists = any('VOYAGER_JWT_SECRET=' in line for line in lines)
            
            if jwt_secret_exists:
                # Update existing JWT secret
                updated_lines = []
                for line in lines:
                    if 'VOYAGER_JWT_SECRET=' in line and not line.strip().startswith('#'):
                        updated_lines.append(f'export VOYAGER_JWT_SECRET="{secret}"\n')
                    else:
                        updated_lines.append(line)
                
                with open(env_file, 'w') as f:
                    f.writelines(updated_lines)
                print(f"✅ Updated existing JWT secret in {env_file}")
            else:
                # Append JWT secret configuration
                with open(env_file, 'a') as f:
                    f.write(f'\n# JWT Secret for Admin Interface\n')
                    f.write(f'export VOYAGER_JWT_SECRET="{secret}"\n')
                    f.write(f'export VOYAGER_JWT_EXPIRE_MINUTES="30"\n')
                print(f"✅ Added JWT secret configuration to {env_file}")
        else:
            # Create new .env file
            with open(env_file, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('# VOYAGER-Trader Environment Variables\n\n')
                f.write('# JWT Secret for Admin Interface\n')
                f.write(f'export VOYAGER_JWT_SECRET="{secret}"\n')
                f.write('export VOYAGER_JWT_EXPIRE_MINUTES="30"\n')
            print(f"✅ Created new .env file with JWT secret: {env_file}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to update {env_file}: {e}")
        return False


def main():
    """Main function to generate and optionally persist JWT secret."""
    print("🔐 VOYAGER-Trader JWT Secret Generator")
    print("=" * 40)
    
    # Generate secret
    secret = generate_jwt_secret()
    print(f"Generated JWT Secret: {secret}")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--env":
        # Update .env file
        env_file = Path(".env")
        if update_env_file(secret, env_file):
            print(f"🎉 JWT secret has been saved to {env_file}")
            print("💡 Run 'source .env' to load the new secret")
        else:
            print("⚠️  Failed to save to .env file")
            print(f"🔧 Manually set: export VOYAGER_JWT_SECRET=\"{secret}\"")
    else:
        print("Manual setup:")
        print(f"export VOYAGER_JWT_SECRET=\"{secret}\"")
        print("export VOYAGER_JWT_EXPIRE_MINUTES=\"30\"")
        print()
        print("Options:")
        print("  --env    Automatically update .env file")
        print()
        print("Example: python scripts/generate_jwt_secret.py --env")


if __name__ == "__main__":
    main()