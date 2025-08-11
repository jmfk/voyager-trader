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
# Development Configuration
# =========================================================

# Set environment (development, testing, production)
export TRADING_ENV="development"

# Enable debug mode
export DEBUG="true"

echo "✅ Environment variables loaded!"
echo "💡 Run 'make config-check' to verify configuration"