"""
Configuration module for Agentic RAG System
Contains all API keys, endpoints, and system settings
"""

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = "3ce80ee7b93f47fe8c77f01b4db0a7c6"
AZURE_OPENAI_ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"

# Model deployments (update these with your deployment names)
GPT4_MINI_MODEL_NAME = "gpt4o"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# Pinecone Configuration
PINECONE_API_KEY = "pcsk_Ey7ZP_EcwGPnAU1ooKcioAFkfiPiTcYAd3tXwQCrveBt5uwYfKyBWC3jS88vX2ZsfcTY8"
PINECONE_INDEX_NAME = "agentic-rag-kb"

# Note: Using Azure OpenAI for self-critique instead of Gemini

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"

print(" Configuration loaded!")