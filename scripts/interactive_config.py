#!/usr/bin/env python3
"""
Interactive configuration setup for VOYAGER-Trader.

This script provides an interactive CLI that guides users through
setting up all required API keys and configurations step-by-step.
"""

import os
import sys
import getpass
import subprocess
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigItem:
    """Represents a configuration item to set up."""

    name: str
    env_var: str
    required: bool = True
    description: str = ""
    setup_instructions: str = ""
    validation_pattern: Optional[str] = None
    is_sensitive: bool = True


class InteractiveConfigSetup:
    """Interactive configuration setup for VOYAGER-Trader."""

    def __init__(self):
        self.config_items = [
            # Core LLM API Keys (at least one required)
            ConfigItem(
                name="OpenAI API Key",
                env_var="OPENAI_API_KEY",
                required=False,  # Either OpenAI or Anthropic needed
                description="API key for OpenAI GPT models (gpt-3.5-turbo, gpt-4, etc.)",
                setup_instructions="""
🔗 How to get your OpenAI API key:
1. Go to: https://platform.openai.com/api-keys
2. Sign up or log in to your OpenAI account
3. Click "Create new secret key"
4. Copy the key (starts with 'sk-')
5. Paste it below when prompted

💡 Note: Keep this key secure and never share it publicly!
""",
                validation_pattern="sk-",
            ),
            ConfigItem(
                name="Anthropic API Key",
                env_var="ANTHROPIC_API_KEY",
                required=False,  # Either OpenAI or Anthropic needed
                description="API key for Anthropic Claude models",
                setup_instructions="""
🔗 How to get your Anthropic API key:
1. Go to: https://console.anthropic.com/settings/keys
2. Sign up or log in to your Anthropic account
3. Click "Create Key"
4. Copy the key (starts with 'sk-ant-')
5. Paste it below when prompted

💡 Note: Keep this key secure and never share it publicly!
""",
                validation_pattern="sk-ant-",
            ),
            # Market Data APIs
            ConfigItem(
                name="Alpha Vantage API Key",
                env_var="ALPHA_VANTAGE_API_KEY",
                required=False,
                description="API key for Alpha Vantage market data (free tier available)",
                setup_instructions="""
🔗 How to get your Alpha Vantage API key:
1. Go to: https://www.alphavantage.co/support/#api-key
2. Enter your email and click "GET FREE API KEY"
3. Check your email for the API key
4. Copy the key (alphanumeric string)
5. Paste it below when prompted

📊 Free tier limits: 25 requests/day, 5 requests/minute
""",
                validation_pattern="",  # No specific pattern
            ),
            # Azure OpenAI (optional)
            ConfigItem(
                name="Azure OpenAI API Key",
                env_var="AZURE_OPENAI_API_KEY",
                required=False,
                description="API key for Azure OpenAI service (enterprise option)",
                setup_instructions="""
🔗 How to get your Azure OpenAI API key:
1. Go to: https://portal.azure.com/
2. Create or navigate to your Azure OpenAI resource
3. Go to "Keys and Endpoint" section
4. Copy either KEY 1 or KEY 2
5. Paste it below when prompted

Note: You'll also need to set the endpoint URL next.
""",
                validation_pattern="",
            ),
            ConfigItem(
                name="Azure OpenAI Endpoint",
                env_var="AZURE_OPENAI_ENDPOINT",
                required=False,
                description="Endpoint URL for Azure OpenAI service",
                setup_instructions="""
🔗 Azure OpenAI Endpoint URL:
1. In your Azure OpenAI resource (Azure Portal)
2. Go to "Keys and Endpoint" section
3. Copy the "Endpoint" URL
4. Should look like: https://your-resource.openai.azure.com/
5. Paste it below when prompted
""",
                validation_pattern="https://",
                is_sensitive=False,
            ),
            # Optional APIs
            ConfigItem(
                name="Google AI API Key",
                env_var="GOOGLE_AI_API_KEY",
                required=False,
                description="API key for Google AI models (optional)",
                setup_instructions="""
🔗 How to get your Google AI API key:
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key
5. Paste it below when prompted
""",
                validation_pattern="",
            ),
            ConfigItem(
                name="Hugging Face API Key",
                env_var="HUGGINGFACE_API_KEY",
                required=False,
                description="API key for Hugging Face models (optional)",
                setup_instructions="""
🔗 How to get your Hugging Face API key:
1. Go to: https://huggingface.co/settings/tokens
2. Sign up or log in to Hugging Face
3. Click "New token"
4. Give it a name and select permissions
5. Copy the token (starts with 'hf_')
6. Paste it below when prompted
""",
                validation_pattern="hf_",
            ),
        ]

        self.env_file_path = Path(".env")
        self.configured_vars = {}

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system("clear" if os.name == "posix" else "cls")

    def print_header(self, title: str):
        """Print a formatted header."""
        self.clear_screen()
        print("🚀 VOYAGER-Trader Configuration Setup")
        print("=" * 50)
        print(f"\n{title}")
        print("-" * len(title))

    def print_progress(self, current: int, total: int):
        """Print progress indicator."""
        progress = "█" * (current * 20 // total) + "░" * (20 - (current * 20 // total))
        print(f"\nProgress: [{progress}] {current}/{total}")

    def get_user_input(
        self, prompt: str, is_sensitive: bool = False, default: str = ""
    ) -> str:
        """Get user input with optional masking for sensitive data."""
        if is_sensitive:
            # Use getpass for sensitive input (hides characters)
            while True:
                try:
                    value = getpass.getpass(f"{prompt}: ")
                    if value.strip():
                        return value.strip()
                    elif default:
                        return default
                    print("❌ Value cannot be empty. Please try again.")
                except KeyboardInterrupt:
                    print("\n\n❌ Configuration cancelled by user.")
                    sys.exit(1)
        else:
            # Regular input for non-sensitive data
            while True:
                try:
                    value = input(f"{prompt}: ")
                    if value.strip():
                        return value.strip()
                    elif default:
                        return default
                    print("❌ Value cannot be empty. Please try again.")
                except KeyboardInterrupt:
                    print("\n\n❌ Configuration cancelled by user.")
                    sys.exit(1)

    def validate_api_key(self, value: str, item: ConfigItem) -> Tuple[bool, str]:
        """Validate an API key format."""
        if not value or not value.strip():
            return False, "Value cannot be empty"

        if item.validation_pattern and not value.startswith(item.validation_pattern):
            return False, f"Expected to start with '{item.validation_pattern}'"

        # Basic length checks
        if len(value) < 10:
            return False, "API key seems too short"

        if " " in value:
            return False, "API key should not contain spaces"

        return True, "Valid"

    def check_existing_env_vars(self) -> Dict[str, str]:
        """Check for existing environment variables."""
        existing = {}
        for item in self.config_items:
            value = os.getenv(item.env_var)
            if value and value.strip():
                existing[item.env_var] = value.strip()
        return existing

    def save_to_env_file(self, vars_dict: Dict[str, str]):
        """Save environment variables to .env file."""
        env_content = []
        env_content.append("#!/bin/bash")
        env_content.append("# VOYAGER-Trader Environment Variables")
        env_content.append("# Generated by interactive configuration")
        env_content.append(f"# Generated on: {os.popen('date').read().strip()}")
        env_content.append("")

        # Core LLM providers
        env_content.append("# =========================================")
        env_content.append("# Core LLM Providers (at least one required)")
        env_content.append("# =========================================")

        llm_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        for var in llm_vars:
            if var in vars_dict:
                env_content.append(f'export {var}="{vars_dict[var]}"')
            else:
                env_content.append(f'# export {var}="your-key-here"')

        # Market data
        env_content.append("")
        env_content.append("# =========================================")
        env_content.append("# Market Data APIs (optional)")
        env_content.append("# =========================================")

        market_vars = ["ALPHA_VANTAGE_API_KEY"]
        for var in market_vars:
            if var in vars_dict:
                env_content.append(f'export {var}="{vars_dict[var]}"')
            else:
                env_content.append(f'# export {var}="your-key-here"')

        # Cloud providers
        env_content.append("")
        env_content.append("# =========================================")
        env_content.append("# Cloud AI Providers (optional)")
        env_content.append("# =========================================")

        cloud_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "GOOGLE_AI_API_KEY",
            "HUGGINGFACE_API_KEY",
        ]
        for var in cloud_vars:
            if var in vars_dict:
                env_content.append(f'export {var}="{vars_dict[var]}"')
            else:
                env_content.append(f'# export {var}="your-key-here"')

        env_content.append("")
        env_content.append("echo '✅ VOYAGER-Trader environment variables loaded!'")
        env_content.append(
            "echo '💡 Run \"make config-check\" to verify configuration'"
        )

        with open(self.env_file_path, "w") as f:
            f.write("\n".join(env_content))

        # Make executable
        os.chmod(self.env_file_path, 0o755)

    def configure_item(self, item: ConfigItem, step: int, total: int) -> Optional[str]:
        """Configure a single item interactively."""
        self.print_header(f"Configure {item.name}")
        self.print_progress(step, total)

        print(f"\n📋 {item.description}")
        print(item.setup_instructions)

        # Check if already exists
        existing_value = os.getenv(item.env_var)
        if existing_value and existing_value.strip():
            masked_value = (
                existing_value[:8] + "..." if item.is_sensitive else existing_value
            )
            print(f"✅ Found existing value: {masked_value}")

            use_existing = input(f"\nUse existing {item.name}? (Y/n): ").lower()
            if use_existing in ["", "y", "yes"]:
                return existing_value.strip()

        # Ask if they want to configure this item
        if not item.required:
            configure = input(
                f"\n❓ Do you want to configure {item.name}? (y/N): "
            ).lower()
            if configure not in ["y", "yes"]:
                print(f"⏭️  Skipping {item.name}")
                input("\nPress Enter to continue...")
                return None

        # Get the API key
        while True:
            print(f"\n🔑 Enter your {item.name}:")
            if item.is_sensitive:
                print("💡 Your input will be hidden for security")

            value = self.get_user_input("API Key", item.is_sensitive)

            # Validate
            is_valid, error_msg = self.validate_api_key(value, item)
            if is_valid:
                print("✅ API key format looks good!")

                # Confirm
                masked_display = value[:8] + "..." if item.is_sensitive else value
                confirm = input(
                    f"✅ Confirm {item.name}: {masked_display}? (Y/n): "
                ).lower()
                if confirm in ["", "y", "yes"]:
                    return value
                else:
                    print("❌ Let's try again...")
            else:
                print(f"❌ {error_msg}")
                retry = input("Try again? (Y/n): ").lower()
                if retry not in ["", "y", "yes"]:
                    return None

    def check_llm_requirements(self, configured: Dict[str, str]) -> bool:
        """Check if at least one LLM provider is configured."""
        has_openai = "OPENAI_API_KEY" in configured and configured["OPENAI_API_KEY"]
        has_anthropic = (
            "ANTHROPIC_API_KEY" in configured and configured["ANTHROPIC_API_KEY"]
        )
        return has_openai or has_anthropic

    def run_interactive_setup(self):
        """Run the complete interactive setup process."""
        self.print_header("Welcome to VOYAGER-Trader Setup!")

        print(
            """
🎯 This setup will guide you through configuring VOYAGER-Trader.

What you'll need:
• At least one LLM API key (OpenAI or Anthropic)  
• Optional: Market data and other AI provider keys

💡 Tips:
• You can skip optional providers
• Your keys will be saved securely in .env file
• Press Ctrl+C anytime to cancel

Ready to start? 🚀
"""
        )

        input("Press Enter to begin...")

        # Check existing configuration
        existing_vars = self.check_existing_env_vars()

        # Configure each item
        total_items = len(self.config_items)
        configured_vars = {}

        for i, item in enumerate(self.config_items, 1):
            value = self.configure_item(item, i, total_items)
            if value:
                configured_vars[item.env_var] = value

        # Include existing vars not reconfigured
        for var, value in existing_vars.items():
            if var not in configured_vars:
                configured_vars[var] = value

        # Check LLM requirements
        if not self.check_llm_requirements(configured_vars):
            self.print_header("⚠️  Missing Required Configuration")
            print(
                """
❌ You need at least one LLM provider configured:
   • OpenAI API Key, OR  
   • Anthropic API Key

VOYAGER-Trader requires an LLM to function.
"""
            )

            retry = input("Go back and configure an LLM provider? (Y/n): ").lower()
            if retry in ["", "y", "yes"]:
                # Retry just the LLM providers
                llm_items = [
                    item
                    for item in self.config_items
                    if item.env_var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
                ]
                for item in llm_items:
                    if item.env_var not in configured_vars:
                        value = self.configure_item(item, 1, 1)
                        if value:
                            configured_vars[item.env_var] = value
                            break

        # Final check
        if not self.check_llm_requirements(configured_vars):
            self.print_header("❌ Configuration Incomplete")
            print("Cannot continue without an LLM provider. Exiting...")
            sys.exit(1)

        # Save configuration
        self.save_to_env_file(configured_vars)

        # Success!
        self.print_header("🎉 Configuration Complete!")

        print(
            f"""
✅ Configuration saved to .env file

📊 Configured providers:
"""
        )

        # Show what was configured
        provider_count = 0
        for item in self.config_items:
            if item.env_var in configured_vars:
                provider_count += 1
                print(f"   ✅ {item.name}")

        print(
            f"""
🔧 Next steps:
1. Load your configuration: source .env
2. Verify setup: make config-check  
3. Run tests: make test
4. Try the system: make run-demo

💡 Your .env file contains your API keys - keep it secure!
"""
        )

        # Ask if they want to load the environment now
        load_now = input("Load environment variables now? (Y/n): ").lower()
        if load_now in ["", "y", "yes"]:
            try:
                # Source the .env file in the current process
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("export "):
                            var_assignment = line[7:].strip()
                            if "=" in var_assignment:
                                var_name, var_value = var_assignment.split("=", 1)
                                var_value = var_value.strip("\"'")
                                os.environ[var_name] = var_value

                print("✅ Environment variables loaded!")

                # Run a quick validation
                print("\n🔍 Running configuration validation...")
                result = subprocess.run(
                    [sys.executable, "scripts/validate_config.py"], capture_output=False
                )

                if result.returncode == 0:
                    print("\n🎉 All set! Your VOYAGER-Trader is ready to use!")

            except Exception as e:
                print(f"⚠️  Could not auto-load environment: {e}")
                print("Please run: source .env")


def main():
    """Main entry point."""
    setup = InteractiveConfigSetup()
    setup.run_interactive_setup()


if __name__ == "__main__":
    main()
