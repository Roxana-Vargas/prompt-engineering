"""Configuration management for the prompt engineering toolkit"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to load streamlit secrets (for Streamlit Cloud deployment)
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get secret from Streamlit secrets or environment variables"""
    # First try environment variable
    value = os.getenv(key, default)
    
    # Then try Streamlit secrets if available
    # Note: Streamlit may raise StreamlitSecretNotFoundError during import/initialization
    # if secrets.toml doesn't exist, so we need to catch it at multiple levels
    try:
        import streamlit as st
    except (ImportError, RuntimeError):
        # Streamlit not available or not in Streamlit context
        return value
    except Exception:
        # Catch StreamlitSecretNotFoundError that might be raised during import
        # This can happen if Streamlit tries to load secrets during initialization
        return value
    
    # Try to access secrets, but catch all errors since StreamlitSecretNotFoundError
    # can be raised when secrets.toml doesn't exist
    try:
        # Use getattr with a default to safely check if secrets attribute exists
        # This avoids triggering the error that might occur with direct access
        secrets_obj = getattr(st, 'secrets', None)
        
        if secrets_obj is not None:
            try:
                # Try to access the specific key from secrets
                # This might still raise StreamlitSecretNotFoundError in some cases
                if hasattr(secrets_obj, '__contains__') and key in secrets_obj:
                    streamlit_value = secrets_obj[key]
                    if streamlit_value:
                        value = streamlit_value
            except (AttributeError, KeyError, TypeError):
                # Key doesn't exist in secrets
                pass
            except Exception:
                # Catch StreamlitSecretNotFoundError and any other Streamlit secret errors
                # This happens when secrets.toml doesn't exist or is not accessible
                # Silently fall back to environment variables
                pass
    except Exception:
        # Catch any errors when checking or accessing st.secrets (including StreamlitSecretNotFoundError)
        # This includes errors that might be raised during attribute access
        # Silently fall back to environment variables
        pass
    
    return value


class Config:
    """Configuration class for API keys and model settings"""
    
    # API Keys (loaded dynamically to support both .env and Streamlit secrets)
    @staticmethod
    def get_openai_key() -> Optional[str]:
        return get_secret("OPENAI_API_KEY")
    
    @staticmethod
    def get_anthropic_key() -> Optional[str]:
        return get_secret("ANTHROPIC_API_KEY")
    
    # Properties for backward compatibility
    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        return self.get_openai_key()
    
    @property
    def ANTHROPIC_API_KEY(self) -> Optional[str]:
        return self.get_anthropic_key()
    
    # Model Configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "1000"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are set"""
        openai_key = cls.get_openai_key()
        anthropic_key = cls.get_anthropic_key()
        
        if not openai_key and not anthropic_key:
            raise ValueError(
                "At least one API key must be set. "
                "Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file "
                "or in Streamlit Cloud Secrets"
            )
        return True

# Create a singleton instance for backward compatibility
_config_instance = Config()

# Make it work like before
Config.OPENAI_API_KEY = property(lambda self: Config.get_openai_key())
Config.ANTHROPIC_API_KEY = property(lambda self: Config.get_anthropic_key())
