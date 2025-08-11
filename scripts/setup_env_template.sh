#!/bin/bash
# VOYAGER-Trader Environment Variables Template
# Copy this file to .env and fill in your actual API keys
# Then run: source .env

# =========================================================
# CRITICAL: At least one LLM provider is required
# =========================================================

# OpenAI API (recommended for best compatibility)
# Get from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-your-openai-key-here"

# Anthropic Claude API (alternative to OpenAI)
# Get from: https://console.anthropic.com/
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"

# =========================================================
# Optional: Market Data APIs
# =========================================================

# Alpha Vantage (free tier: 25 requests/day)
# Get from: https://www.alphavantage.co/support/#api-key
export ALPHA_VANTAGE_API_KEY="your-alpha-vantage-key"

# =========================================================
# Optional: Azure OpenAI (enterprise alternative)
# =========================================================

# Azure OpenAI Service
# Get from: Azure Portal > Your OpenAI Resource
export AZURE_OPENAI_API_KEY="your-azure-openai-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# =========================================================
# Optional: Additional AI Providers
# =========================================================

# Google AI Studio
# Get from: https://makersuite.google.com/app/apikey
export GOOGLE_AI_API_KEY="your-google-ai-key"

# Hugging Face
# Get from: https://huggingface.co/settings/tokens
export HUGGINGFACE_API_KEY="hf_your-huggingface-token"

# =========================================================
# Admin Interface Security
# =========================================================

# JWT Secret for admin interface authentication
# Generate a secure secret or set your own
export VOYAGER_JWT_SECRET="$(openssl rand -base64 32 2>/dev/null || python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

# JWT token expiration time in minutes
export VOYAGER_JWT_EXPIRE_MINUTES="30"

# Admin credentials (CHANGE FOR PRODUCTION!)
export VOYAGER_ADMIN_PASSWORD="your-secure-admin-password"
# Or set pre-hashed password (takes precedence):
# export VOYAGER_ADMIN_PASSWORD_HASH="$2b$12$your-bcrypt-hash-here"

# =========================================================
# Development Configuration
# =========================================================

# Set environment (development, testing, production)
export TRADING_ENV="development"

# Enable debug mode
export DEBUG="true"

echo "✅ Environment variables loaded!"
echo "💡 Run 'make config-check' to verify configuration"
echo "🔐 JWT Secret configured for admin interface security"