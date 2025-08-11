#!/usr/bin/env python3
"""
Configuration validation script for VOYAGER-Trader.

This script checks for required API keys and configurations,
provides helpful guidance for missing items, and validates
the system is properly configured.
"""

import os
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigItem:
    """Represents a configuration item to validate."""

    name: str
    env_var: str
    required: bool = True
    description: str = ""
    setup_instructions: str = ""


class ConfigValidator:
    """Validates VOYAGER-Trader configuration."""

    def __init__(self):
        self.config_items = [
            # Core LLM API Keys (at least one required)
            ConfigItem(
                name="OpenAI API Key",
                env_var="OPENAI_API_KEY",
                required=False,  # Either OpenAI or Anthropic needed
                description="API key for OpenAI GPT models (gpt-3.5-turbo, gpt-4, etc.)",
                setup_instructions="""
1. Sign up at https://platform.openai.com/
2. Navigate to API Keys section
3. Create a new API key
4. Export: export OPENAI_API_KEY="sk-..."
""",
            ),
            ConfigItem(
                name="Anthropic API Key",
                env_var="ANTHROPIC_API_KEY",
                required=False,  # Either OpenAI or Anthropic needed
                description="API key for Anthropic Claude models",
                setup_instructions="""
1. Sign up at https://console.anthropic.com/
2. Navigate to API Keys section
3. Create a new API key
4. Export: export ANTHROPIC_API_KEY="sk-ant-..."
""",
            ),
            # Market Data APIs
            ConfigItem(
                name="Alpha Vantage API Key",
                env_var="ALPHA_VANTAGE_API_KEY",
                required=False,
                description="API key for Alpha Vantage market data (optional)",
                setup_instructions="""
1. Sign up at https://www.alphavantage.co/support/#api-key
2. Get your free API key
3. Export: export ALPHA_VANTAGE_API_KEY="your_key_here"
Note: Free tier has 25 requests/day, 5 requests/minute limits
""",
            ),
            # Azure OpenAI (optional)
            ConfigItem(
                name="Azure OpenAI API Key",
                env_var="AZURE_OPENAI_API_KEY",
                required=False,
                description="API key for Azure OpenAI service (optional)",
                setup_instructions="""
1. Create Azure OpenAI resource in Azure Portal
2. Get API key and endpoint from resource
3. Export: export AZURE_OPENAI_API_KEY="your_key_here"
4. Export: export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
""",
            ),
            ConfigItem(
                name="Azure OpenAI Endpoint",
                env_var="AZURE_OPENAI_ENDPOINT",
                required=False,
                description="Endpoint URL for Azure OpenAI service",
                setup_instructions="Set along with AZURE_OPENAI_API_KEY",
            ),
            # Optional APIs
            ConfigItem(
                name="Google AI API Key",
                env_var="GOOGLE_AI_API_KEY",
                required=False,
                description="API key for Google AI models (optional)",
                setup_instructions="""
1. Visit Google AI Studio: https://makersuite.google.com/app/apikey
2. Create API key
3. Export: export GOOGLE_AI_API_KEY="your_key_here"
""",
            ),
            ConfigItem(
                name="Hugging Face API Key",
                env_var="HUGGINGFACE_API_KEY",
                required=False,
                description="API key for Hugging Face models (optional)",
                setup_instructions="""
1. Sign up at https://huggingface.co/
2. Go to Settings > Access Tokens
3. Create new token
4. Export: export HUGGINGFACE_API_KEY="hf_..."
""",
            ),
        ]

    def check_python_version(self) -> Tuple[bool, str]:
        """Check if Python version meets requirements."""
        version = sys.version_info
        if version >= (3, 12):
            return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
        else:
            return (
                False,
                f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.12+)",
            )

    def check_project_structure(self) -> Tuple[bool, List[str]]:
        """Check if required project directories exist."""
        required_paths = [
            "src/voyager_trader",
            "tests",
            "requirements.txt",
        ]

        missing = []
        for path in required_paths:
            if not Path(path).exists():
                missing.append(path)

        return len(missing) == 0, missing

    def check_environment_variables(self) -> Tuple[Dict[str, bool], List[str]]:
        """Check which environment variables are set."""
        results = {}
        missing_required = []

        for item in self.config_items:
            value = os.getenv(item.env_var)
            is_set = value is not None and value.strip() != ""
            results[item.env_var] = is_set

            if item.required and not is_set:
                missing_required.append(item.name)

        return results, missing_required

    def check_llm_availability(self, env_results: Dict[str, bool]) -> Tuple[bool, str]:
        """Check if at least one LLM provider is available."""
        openai_available = env_results.get("OPENAI_API_KEY", False)
        anthropic_available = env_results.get("ANTHROPIC_API_KEY", False)

        if openai_available or anthropic_available:
            providers = []
            if openai_available:
                providers.append("OpenAI")
            if anthropic_available:
                providers.append("Anthropic")
            return True, f"✅ LLM providers available: {', '.join(providers)}"
        else:
            return (
                False,
                "❌ No LLM provider configured (need OPENAI_API_KEY or ANTHROPIC_API_KEY)",
            )

    def print_setup_guidance(self, env_results: Dict[str, bool]):
        """Print setup guidance for missing configurations."""
        print("\n🔧 Configuration Setup Guide")
        print("=" * 50)

        # Check for critical missing items
        has_llm = env_results.get("OPENAI_API_KEY", False) or env_results.get(
            "ANTHROPIC_API_KEY", False
        )

        if not has_llm:
            print("\n❗ CRITICAL: At least one LLM provider is required")
            print("-" * 40)

            for item in self.config_items:
                if item.env_var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
                    if not env_results.get(item.env_var, False):
                        print(f"\n🔑 {item.name}")
                        print(f"   Description: {item.description}")
                        print(f"   Setup:{item.setup_instructions}")

        # Show optional configurations
        print("\n📋 Optional Configurations")
        print("-" * 30)

        optional_items = [
            item
            for item in self.config_items
            if not item.required
            and item.env_var not in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        ]

        for item in optional_items:
            status = (
                "✅ Configured"
                if env_results.get(item.env_var, False)
                else "⚪ Not configured"
            )
            print(f"\n{status} {item.name}")
            if not env_results.get(item.env_var, False) and item.description:
                print(f"   {item.description}")
                if item.setup_instructions.strip():
                    print(f"   Setup:{item.setup_instructions}")

    def validate(self) -> bool:
        """Run complete configuration validation."""
        print("🔍 VOYAGER-Trader Configuration Validation")
        print("=" * 50)

        all_good = True

        # Check Python version
        python_ok, python_msg = self.check_python_version()
        print(f"\n{python_msg}")
        if not python_ok:
            all_good = False

        # Check project structure
        structure_ok, missing_paths = self.check_project_structure()
        if structure_ok:
            print("✅ Project structure valid")
        else:
            print(f"❌ Missing project files: {', '.join(missing_paths)}")
            all_good = False

        # Check environment variables
        env_results, missing_required = self.check_environment_variables()

        # Check LLM availability
        llm_ok, llm_msg = self.check_llm_availability(env_results)
        print(f"\n{llm_msg}")
        if not llm_ok:
            all_good = False

        # Print environment variable status
        print(f"\n📊 Environment Variables Status")
        print("-" * 30)

        for item in self.config_items:
            is_set = env_results.get(item.env_var, False)
            status = "✅" if is_set else ("❌" if item.required else "⚪")
            required_text = " (required)" if item.required else " (optional)"

            print(f"{status} {item.name}{required_text}")

        # Show setup guidance if needed
        if not all_good or not all(
            env_results.get(item.env_var, False)
            for item in self.config_items
            if item.required
        ):
            self.print_setup_guidance(env_results)

        # Final status
        print(
            f"\n{'🎉 Configuration Complete!' if all_good else '⚠️  Configuration Issues Found'}"
        )
        print("=" * 50)

        if all_good:
            print("✅ System is ready! You can now run:")
            print("   make test       # Run tests")
            print("   make run-demo   # Try the system")
        else:
            print("❌ Please fix configuration issues above before continuing.")
            print("💡 After setting environment variables, run 'make configure' again.")

        return all_good


def main():
    """Main entry point."""
    validator = ConfigValidator()
    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
