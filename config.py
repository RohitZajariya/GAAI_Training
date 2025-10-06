"""
Configuration module for Agentic RAG System
Contains all API keys, endpoints, and system settings
"""

# =============================================================================
# 🔑 CONFIGURATION - UPDATE WITH YOUR API KEYS
# =============================================================================

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = "your_azure_openai_api_key_here"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

# Model deployments (update these with your deployment names)
GPT4_MINI_MODEL_NAME = "gpt-4-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# Pinecone Configuration
PINECONE_API_KEY = "pcsk_Ey7ZP_EcwGPnAU1ooKcioAFkfiPiTcYAd3tXwQCrveBt5uwYfKyBWC3jS88vX2ZsfcTY8"
PINECONE_INDEX_NAME = "agentic-rag-kb"

# Note: Using Azure OpenAI for self-critique instead of Gemini

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Debug Configuration
DEBUG_MODE = True
LOG_LEVEL = "INFO"

print("⚙️ Configuration loaded!")

# Validation function
def validate_config():
    """Validate configuration settings"""
    issues = []
    
    if AZURE_OPENAI_API_KEY == "your_azure_openai_api_key_here":
        issues.append("Azure OpenAI API Key not configured")
    
    if AZURE_OPENAI_ENDPOINT == "https://your-resource.openai.azure.com/":
        issues.append("Azure OpenAI Endpoint not configured")
    
    if PINECONE_API_KEY == "your_pinecone_api_key_here":
        issues.append("Pinecone API Key not configured")
    
    # Note: Gemini API key check removed since we are using Azure OpenAI for critique
    
    if issues:
        print("❌ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All configurations are set!")
        return True
