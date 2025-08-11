#!/usr/bin/env python3
"""
Test the interactive configuration system without user input.
This simulates the interactive setup for testing purposes.
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from interactive_config import InteractiveConfigSetup, ConfigItem


class TestInteractiveConfig(InteractiveConfigSetup):
    """Test version of interactive config that simulates user input."""
    
    def __init__(self, test_responses=None):
        super().__init__()
        # Test responses for simulation
        self.test_responses = test_responses or {}
        self.current_step = 0
    
    def clear_screen(self):
        """Don't clear screen during testing."""
        pass
    
    def get_user_input(self, prompt: str, is_sensitive: bool = False, default: str = "") -> str:
        """Simulate user input for testing."""
        self.current_step += 1
        
        # Return test response if available
        if prompt in self.test_responses:
            response = self.test_responses[prompt]
            print(f"[TEST] Simulating input for '{prompt}': {'***' if is_sensitive else response}")
            return response
        
        # Default responses
        if "API Key" in prompt:
            if "OpenAI" in prompt:
                print("[TEST] Simulating OpenAI API key input")
                return "sk-test-openai-key-1234567890abcdef"
            elif "Anthropic" in prompt:
                print("[TEST] Simulating Anthropic API key input")
                return "sk-ant-test-anthropic-key-1234567890"
        
        return default
    
    def configure_item(self, item: ConfigItem, step: int, total: int) -> str:
        """Simplified configure item for testing."""
        print(f"\n[TEST] Configuring {item.name} ({step}/{total})")
        
        # For LLM providers, always configure the first one (OpenAI)
        if item.env_var == "OPENAI_API_KEY":
            print("[TEST] Setting up OpenAI API key")
            return "sk-test-openai-key-1234567890abcdef"
        
        # Skip optional items for testing
        if not item.required:
            print(f"[TEST] Skipping optional {item.name}")
            return None
        
        return None
    
    def run_test_setup(self):
        """Run a test setup simulation."""
        print("🧪 Testing Interactive Configuration Setup")
        print("=" * 50)
        
        # Test validation functions
        print("\n1. Testing validation functions...")
        
        # Test API key validation
        test_cases = [
            ("sk-test123", ConfigItem("Test", "TEST", validation_pattern="sk-")),
            ("sk-ant-test123", ConfigItem("Test", "TEST", validation_pattern="sk-ant-")),
            ("invalid", ConfigItem("Test", "TEST", validation_pattern="sk-")),
            ("", ConfigItem("Test", "TEST")),
            ("short", ConfigItem("Test", "TEST")),
        ]
        
        for test_key, item in test_cases:
            is_valid, msg = self.validate_api_key(test_key, item)
            status = "✅" if is_valid else "❌"
            print(f"   {status} '{test_key}': {msg}")
        
        # Test LLM requirement checking
        print("\n2. Testing LLM requirement checking...")
        test_configs = [
            {"OPENAI_API_KEY": "sk-test123"},
            {"ANTHROPIC_API_KEY": "sk-ant-test123"},
            {"ALPHA_VANTAGE_API_KEY": "test123"},  # Should fail
            {},  # Should fail
            {"OPENAI_API_KEY": "sk-test123", "ANTHROPIC_API_KEY": "sk-ant-test123"},  # Should pass
        ]
        
        for i, config in enumerate(test_configs, 1):
            has_llm = self.check_llm_requirements(config)
            status = "✅" if has_llm else "❌"
            print(f"   {status} Config {i}: {config}")
        
        # Test environment file creation
        print("\n3. Testing .env file generation...")
        test_vars = {
            "OPENAI_API_KEY": "sk-test-key-12345",
            "ALPHA_VANTAGE_API_KEY": "test-alpha-key"
        }
        
        # Save to test file
        test_env_path = Path(".env.test")
        original_env_path = self.env_file_path
        self.env_file_path = test_env_path
        
        try:
            self.save_to_env_file(test_vars)
            if test_env_path.exists():
                print("   ✅ .env file created successfully")
                with open(test_env_path) as f:
                    content = f.read()
                    if "OPENAI_API_KEY" in content and "sk-test-key-12345" in content:
                        print("   ✅ API keys properly saved")
                    else:
                        print("   ❌ API keys not found in file")
            else:
                print("   ❌ .env file not created")
        finally:
            # Cleanup
            if test_env_path.exists():
                test_env_path.unlink()
            self.env_file_path = original_env_path
        
        print("\n🎉 Interactive configuration tests completed!")
        return True


def main():
    """Run the tests."""
    tester = TestInteractiveConfig()
    success = tester.run_test_setup()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()