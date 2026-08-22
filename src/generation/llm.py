"""
LLM Manager to instantiate multi-provider LLM clients (Groq / Gemini)
"""
from langchain_core.language_models.chat_models import BaseChatModel
from src.core.config import get_settings
from src.core.logger import get_logger
import os

logger = get_logger("llm_manager")

class LLMManager:
    """Factory for LangChain LLM Clients."""
    
    @staticmethod
    def get_llm() -> BaseChatModel:
        settings = get_settings()
        provider = settings.llm_provider.lower()
        
        if provider == "groq":
            from langchain_groq import ChatGroq
            api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set in environment or config.")
            
            logger.info(f"Initializing Groq LLM: {settings.llm_model}")
            return ChatGroq(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_output_tokens,
                api_key=api_key
            )
            
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set in environment or config.")
                
            logger.info(f"Initializing Gemini LLM: {settings.llm_model}")
            return ChatGoogleGenerativeAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_output_tokens,
                google_api_key=api_key
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
