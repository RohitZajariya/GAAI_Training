"""
Data Loader module for Agentic RAG System
Handles loading and processing of knowledge base data
"""

import json
from typing import List, Dict, Any

def load_kb_data(file_path: str = 'self_critique_loop_dataset.json') -> List[Dict[str, Any]]:
    """
    Load knowledge base data from JSON file
    
    Args:
        file_path: Path to the JSON file containing KB data
        
    Returns:
        List of knowledge base entries
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"✅ Loaded {len(data)} KB entries from {file_path}")
            
            # Display sample entry for verification
            print("\n📋 Sample KB Entry:")
            sample = data[0]
            for key, value in sample.items():
                if key == 'answer_snippet':
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
            
            return data
            
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please ensure the self_critique_loop_dataset.json file is in the current directory")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return []
    except Exception as e:
        print(f"❌ Failed to load KB data: {e}")
        return []

def format_kb_entry_for_embedding(entry: Dict[str, Any]) -> str:
    """
    Format KB entry for embedding generation
    
    Args:
        entry: KB entry dictionary
        
    Returns:
        Formatted text for embedding
    """
    return f"{entry['question']} {entry['answer_snippet']}"

def get_kb_entry_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of a KB entry for logging/debugging
    
    Args:
        entry: KB entry dictionary
        
    Returns:
        Summary dictionary
    """
    return {
        "doc_id": entry['doc_id'],
        "question": entry['question'],
        "source": entry['source'],
        "confidence": entry['confidence_indicator'],
        "last_updated": entry['last_updated'],
        "snippet_preview": entry['answer_snippet'][:100] + "..." if len(entry['answer_snippet']) > 100 else entry['answer_snippet']
    }

# Test function
def test_data_loader():
    """Test the data loader functionality"""
    print("\n🧪 Testing Data Loader...")
    
    # Test loading data
    data = load_kb_data()
    
    if data:
        print(f"✅ Successfully loaded {len(data)} entries")
        
        # Test formatting
        sample_entry = data[0]
        formatted_text = format_kb_entry_for_embedding(sample_entry)
        print(f"✅ Embedding text formatted: {len(formatted_text)} characters")
        
        # Test summary
        summary = get_kb_entry_summary(sample_entry)
        print(f"✅ Entry summary created: {summary}")
        
        return True
    else:
        print("❌ Data loading failed")
        return False

if __name__ == "__main__":
    test_data_loader()
