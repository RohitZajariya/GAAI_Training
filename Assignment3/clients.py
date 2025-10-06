"""
Client initialization module for Agentic RAG System
Handles setup of Azure OpenAI, Pinecone, and MLflow clients
"""

from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import ChatRequestMessage, ChatRole, UserMessage, SystemMessage
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from pinecone import Pinecone
import mlflow
from typing import Optional

from config import *

class ClientManager:
    """Manages all external service clients"""
    
    def __init__(self):
        self.azure_openai_client: Optional[ChatCompletionsClient] = None
        self.embedding_client: Optional[EmbeddingsClient] = None
        self.pc_client: Optional[Pinecone] = None

        self.pinecone_index = None
        
    def initialize_azure_openai(self) -> bool:
        """Initialize Azure OpenAI clients"""
        try:
    

            self.azure_openai_client = AzureOpenAI(api_key = AZURE_OPENAI_API_KEY, api_version = AZURE_OPENAI_API_VERSION, azure_endpoint = AZURE_OPENAI_ENDPOINT)

            self.embedding_client = AzureOpenAI(api_key = AZURE_OPENAI_API_KEY, api_version = AZURE_OPENAI_API_VERSION, azure_endpoint = AZURE_OPENAI_ENDPOINT)
       
            print("Azure OpenAI clients initialized")
            return True
        except Exception as e:
            print(f" Azure OpenAI setup failed: {e}")
            return False
    
    def initialize_pinecone(self) -> bool:
        """Initialize Pinecone client"""
        try:
            self.pc_client = Pinecone(api_key=PINECONE_API_KEY)
            print(" Pinecone client initialized")
            
            # List existing indexes
            existing_indexes = self.pc_client.list_indexes()
            print(f"Available indexes: {[idx.name for idx in existing_indexes]}")
            return True
        except Exception as e:
            print(f" Pinecone setup failed: {e}")
            return False
    

    
    def initialize_all(self) -> bool:
        """Initialize all clients"""
        print("Initializing all service clients...")
        
        results = [
            self.initialize_azure_openai(),
            self.initialize_pinecone()
        ]
        
        success_count = sum(results)
        total_clients = len(results)
        
        print(f"\n Initialization Summary: {success_count}/{total_clients} clients ready")
        
        if success_count == total_clients:
            print("All clients initialized successfully!")
            return True
        else:
            print(" Some clients failed to initialize")
            return False
    
    def create_pinecone_index(self, kb_data: list) -> bool:
        """Create and populate Pinecone index with KB data"""
        if not self.pc_client or not self.embedding_client or not kb_data:
            print(" Missing required clients or data for index creation")
            return False
        
        try:
            # Check existing indexes
            existing_indexes = [idx.name for idx in self.pc_client.list_indexes()]
            
            if PINECONE_INDEX_NAME in existing_indexes:
                print(f"Index '{PINECONE_INDEX_NAME}' already exists")
                self.pinecone_index = self.pc_client.Index(PINECONE_INDEX_NAME)
            else:
                # Get embedding dimension
                test_response = self.embedding_client.embeddings.create(input = ["dimension test"], model=EMBEDDING_MODEL_NAME)
                dimension = len(test_response.data[0].embedding)
                
                print(f"Creating index '{PINECONE_INDEX_NAME}' with dimension {dimension}")
                
                self.pc_client.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=dimension,
                    metric='cosine',
                    spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}}
                )
                
                print(f" Index created successfully")
                import time
                time.sleep(10)  # Wait for index
                self.pinecone_index = self.pc_client.Index(PINECONE_INDEX_NAME)
            
            # Check if index needs population
            stats = self.pinecone_index.describe_index_stats()
            if stats['total_vector_count'] == 0:
                print("Populating index with KB data...")
                self._populate_index(kb_data)
            else:
                print(f"Index contains {stats['total_vector_count']} vectors")
            
            return True
            
        except Exception as e:
            print(f"Failed to setup index: {e}")
            return False
    
    def _populate_index(self, kb_data: list):
        """Populate Pinecone index with KB embeddings"""
        from data_loader import format_kb_entry_for_embedding
        
        vectors_to_upsert = []
        
        for entry in kb_data:
            try:
                # Create embedding text
                text_to_embed = format_kb_entry_for_embedding(entry)

                # print("Text to embed completed")
                
                # Generate embedding
                embedding_response = self.embedding_client.embeddings.create(input = [text_to_embed], model=EMBEDDING_MODEL_NAME)

                # print("Embedding generation completed")
                embedding = embedding_response.data[0].embedding
                
                # Prepare vector data
                vector_data = {
                    'id': entry['doc_id'],
                    'values': embedding,
                    'metadata': {
                        'question': entry['question'],
                        'answer_snippet': entry['answer_snippet'],
                        'source': entry['source'],
                        'confidence_indicator': entry['confidence_indicator'],
                        'last_updated': entry['last_updated']
                    }
                }
                vectors_to_upsert.append(vector_data)
                
            except Exception as e:
                print(f" Failed to process {entry['doc_id']}: {e}")
        
        # Upsert vectors sorted by doc_id for consistency
        vectors_to_upsert.sort(key=lambda x: x['id'])
        
        try:
            self.pinecone_index.upsert(vectors=vectors_to_upsert)
            print(f" Successfully upserted {len(vectors_to_upsert)} vectors")
        except Exception as e:
            print(f" Failed to upsert: {e}")
    
    def search_pinecone(self, query: str, top_k: int = 5) -> list:
        """Search Pinecone index"""
        if not self.pinecone_index or not self.embedding_client:
            print(" Pinecone index or embedding client not initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_client.embeddings.create(input = [query], model=EMBEDDING_MODEL_NAME)
            
            # Search Pinecone
            search_results = self.pinecone_index.query(
                vector=query_embedding.data[0].embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            results = []
            for match in search_results.matches:
                doc_data = {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                }
                results.append(doc_data)
            
            return results
            
        except Exception as e:
            print(f" Pinecone search failed: {e}")
            return []

# Global client manager instance
client_manager = ClientManager()

def initialize_system(kb_data: list) -> bool:
    """Initialize the entire system"""
    success = client_manager.initialize_all()
    if success:
        success = client_manager.create_pinecone_index(kb_data)
    return success

def test_clients():
    """Test client initialization"""
    print("\nTesting Client Initialization...")
    
    from data_loader import load_kb_data
    
    # Load test data
    kb_data = load_kb_data()
    if not kb_data:
        print(" Cannot test without KB data")
        return False
    
    # Test initialization
    success = initialize_system(kb_data)
    
    if success:
        print(" All clients and system initialized successfully!")
        
        # Test search functionality
        test_results = client_manager.search_pinecone("test query", top_k=3)
        print(f" Search test: Found {len(test_results)} results")
        
        return True
    else:
        print(" System initialization failed")
        return False

if __name__ == "__main__":
    # validate_config()
    test_clients()
