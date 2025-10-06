"""
RAG Nodes module for Agentic RAG System
Contains all LangGraph workflow nodes implementation
"""

from typing import List, Dict, Any, TypedDict
from clients import client_manager
from config import GPT4_MINI_MODEL_NAME, EMBEDDING_MODEL_NAME
from azure.ai.inference.models import SystemMessage, UserMessage

# Define the state for our agentic RAG system
class AgenticRAGState(TypedDict):
    query: str
    retrieved_docs: List[Dict[str, Any]]
    initial_answer: str
    critique_result: str
    refinement_needed: bool
    refined_answer: str

def retriever_node(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 1: Retriever - Fetch top 5 KB snippets
    """
    

    print(f"\nRETRIEVER: Searching for '{state['query'][:50]}...'")


    # Use client manager to search Pinecone
    retrieved_docs = client_manager.search_pinecone(state['query'], top_k=5)
    
    # Log retrieved documents

    for doc in retrieved_docs:
        print(f" {doc['id']}: {doc['score']:.4f} - {doc['metadata'].get('question', 'N/A')[:50]}...")
    
    state['retrieved_docs'] = retrieved_docs
    

    print(f" Retrieved {len(retrieved_docs)} documents")
    
    return state

        # return state

def llm_answer_node(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 2: LLM Answer Generation using Azure GPT-4 mini
    """
    

    print("\nLLM ANSWER: Generating initial answer")
    
    try:
        # Prepare context from retrieved docs
        context_parts = []
        for doc in state['retrieved_docs']:
            doc_id = doc['id']
            question = doc['metadata'].get('question', '')
            answer_snippet = doc['metadata'].get('answer_snippet', '')
            context_parts.append(f"[{doc_id}] {question}: {answer_snippet}")
        
        context = "\n\n".join(context_parts)
        
        # Create system prompt
        system_prompt = """You are a helpful assistant that answers questions using provided knowledge base snippets.
Always cite sources using [KBxxx] format where xxx is the document ID.
Provide comprehensive answers based only on the given information."""
        
        user_prompt = f"""Knowledge Base Context:\n{context}\n\nQuestion: {state['query']}\n\nPlease provide a comprehensive answer with [KBxxx] citations."""
        
        # Call Azure OpenAI
        response = client_manager.azure_openai_client.chat.completions.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ],
            model=GPT4_MINI_MODEL_NAME,
            temperature=0
        )
        
        initial_answer = response.choices[0].message.content
        state['initial_answer'] = initial_answer
        
        
        print(f"Generated answer ({len(initial_answer)} chars)")
        print(f"Preview: {initial_answer[:150]}...")
        
        return state
        
    except Exception as e:
        print(f" LLM Answer failed: {e}")
        state['initial_answer'] = "Error: Could not generate answer."
        return state

def self_critique_node(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 3: Self-Critique using Azure OpenAI (replaces Gemini)
    """
    
    
    print("\n SELF-CRITIQUE: Analyzing answer completeness")
    
    try:
        critique_prompt = f"""Evaluate if this answer is COMPLETE or needs REFINEMENT:

Question: {state['query']}

Answer: {state['initial_answer']}

Respond with ONLY one word:
- "COMPLETE" if answer fully addresses question
- "REFINE" if needs additional information

Verdict:"""
        
        response = client_manager.azure_openai_client.chat.completions.create(
            messages=[
                SystemMessage(content="You are an objective critique assistant. Evaluate answers based on completeness and accuracy."),
                UserMessage(content=critique_prompt)
            ],
            model=GPT4_MINI_MODEL_NAME,
            temperature=0
        )
        
        critique_result = response.choices[0].message.content.strip().upper()
        
        # Clean up the response to ensure it's either COMPLETE or REFINE
        if "COMPLETE" in critique_result:
            critique_result = "COMPLETE"
        elif "REFINE" in critique_result:
            critique_result = "REFINE"
        else:
            # Default to COMPLETE if unclear response
            critique_result = "COMPLETE"
        
        refinement_needed = critique_result == "REFINE"
        
        state['critique_result'] = critique_result
        state['refinement_needed'] = refinement_needed
        
        
        print(f" Critique (Azure): {critique_result}")
        print(f"Refinement needed: {refinement_needed}")
        
        return state
        
    except Exception as e:
        print(f" Critique failed: {e}")
        state['critique_result'] = "COMPLETE"
        state['refinement_needed'] = False
        return state

def self_critique_azure_node(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 3 Alternative: Self-Critique using Azure OpenAI (legacy fallback - now primary)
    """
    
    
    print("\n SELF-CRITIQUE (Azure): Analyzing answer")
    
    try:
        critique_prompt = f"""Evaluate if answer is COMPLETE or REFINE:\n\nQuestion: {state['query']}\n\nAnswer: {state['initial_answer']}\n\nRespond with ONLY one word: COMPLETE or REFINE\n\nVerdict:"""
        
        response = client_manager.azure_openai_client.chat.completions.create(
            messages=[
                SystemMessage(content="Evaluate answers objectively."),
                UserMessage(content=critique_prompt)
            ],
            model=GPT4_MINI_MODEL_NAME,
            temperature=0
        )
        
        critique_result = response.choices[0].message.content.strip().upper()
        refinement_needed = critique_result == "REFINE"
        
        state['critique_result'] = critique_result
        state['refinement_needed'] = refinement_needed
        
    
        print(f" Critique (Azure): {critique_result}")
        print(f" Refinement needed: {refinement_needed}")
        
        return state
        
    except Exception as e:
        print(f" Critique (Azure) failed: {e}")
        state['critique_result'] = "COMPLETE"
        state['refinement_needed'] = False
        return state

def refinement_node(state: AgenticRAGState) -> AgenticRAGState:
    """
    Node 4: Refinement - Retrieve additional snippet and regenerate answer
    """
    

    print("\n REFINEMENT: Getting additional snippet")
    
    try:
        # Get 1 more snippet (6th result)
        additional_results = client_manager.search_pinecone(state['query'], top_k=6)
        
        # Find a new snippet not in the original retrieval
        existing_ids = {doc['id'] for doc in state['retrieved_docs']}
        new_snippet = None
        
        for result in additional_results:
            if result['id'] not in existing_ids:
                new_snippet = result
                break
        
        if not new_snippet:
            
            print("No new snippet found, using original answer")
            state['refined_answer'] = state['initial_answer']
            return state
        
        # Combine original docs with new snippet
        all_docs = state['retrieved_docs'] + [new_snippet]
        
        # Prepare enhanced context
        context_parts = []
        for doc in all_docs:
            doc_id = doc['id']
            question = doc['metadata'].get('question', '')
            answer_snippet = doc['metadata'].get('answer_snippet', '')
            context_parts.append(f"[{doc_id}] {question}: {answer_snippet}")
        
        context = "\n\n".join(context_parts)
        
        # Generate refined answer
        system_prompt = """You are a helpful assistant that answers questions using provided knowledge base snippets.
Always cite sources using [KBxxx] format. Provide comprehensive answers based only on the given information.
This is a refined answer, so ensure it addresses the question completely."""
        
        user_prompt = f"""Enhanced Knowledge Base Context:\n{context}\n\nOriginal Question: {state['query']}\n\nOriginal Answer: {state['initial_answer']}\n\nPlease provide a COMPLETE and ENHANCED answer using all available information with [KBxxx] citations."""
        
        response = client_manager.azure_openai_client.chat.completions.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ],
            model=GPT4_MINI_MODEL_NAME,
            temperature=0
        )
        
        refined_answer = response.choices[0].message.content
        state['refined_answer'] = refined_answer
        
        
        print(f" Generated refined answer ({len(refined_answer)} chars)")
        print(f"Preview: {refined_answer[:150]}...")
        
        return state
        
    except Exception as e:
        print(f" Refinement failed: {e}")
        state['refined_answer'] = state['initial_answer']
        return state

def run_agentic_rag_pipeline(query: str) -> Dict[str, Any]:
    """
    Run the complete agentic RAG pipeline
    
    Args:
        query: User question
        
    Returns:
        Complete pipeline result
    """
    
    
    print(f"\nStarting Agentic RAG Pipeline for: '{query}'")

    # Initialize state
    state = AgenticRAGState(
        query=query,
        retrieved_docs=[],
        initial_answer="",
        critique_result="",
        refinement_needed=False,
        refined_answer=""
    )
    
    try:
        # Step 1: Retrieve documents
        state = retriever_node(state)
        
        if not state['retrieved_docs']:
            return {
                "error": "No documents retrieved",
                "query": query,
                "final_answer": "I couldn't find relevant information to answer your question."
            }
        
        # Step 2: Generate initial answer
        state = llm_answer_node(state)
        
        # Step 3: Self-critique (using Azure OpenAI)
        state = self_critique_node(state)
        
        # Step 4: Refinement if needed
        if state['refinement_needed']:
            state = refinement_node(state)
        else:
            state['refined_answer'] = state['initial_answer']
        
        return {
            "query": query,
            "retrieved_docs": state['retrieved_docs'],
            "initial_answer": state['initial_answer'],
            "critique_result": state['critique_result'],
            "refinement_needed": state['refinement_needed'],
            "refined_answer": state['refined_answer'],
            "final_answer": state['refined_answer']
        }
        
    except Exception as e:
        print(f" Pipeline failed: {e}")
        return {
            "error": str(e),
            "query": query,
            "final_answer": "An error occurred while processing your question."
        }

def test_nodes():
    """Test the individual nodes"""
    print("\nTesting RAG Nodes...")
    
    # Test with a sample query
    test_query = "What are best practices for caching?"
    
    try:
        # Test retriever node
        test_state = AgenticRAGState(
            query=test_query,
            retrieved_docs=[],
            initial_answer="",
            critique_result="",
            refinement_needed=False,
            refined_answer=""
        )
        
        # Test each node
        print("1. Testing retriever node...")
        test_state = retriever_node(test_state)
        print(f"   Retrieved {len(test_state['retrieved_docs'])} docs")
        
        if test_state['retrieved_docs']:
            print("2. Testing LLM answer node...")
            test_state = llm_answer_node(test_state)
            print(f"   Generated answer: {len(test_state['initial_answer'])} chars")
            
            print("3. Testing self-critique node...")
            test_state = self_critique_node(test_state)
            print(f"   Critique result: {test_state['critique_result']}")
            
            print("4. Testing refinement node...")
            if test_state['refinement_needed']:
                test_state = refinement_node(test_state)
                print(f"   Refined answer: {len(test_state['refined_answer'])} chars")
            else:
                print("   Refinement not needed")
        
        print(" All nodes tested successfully!")
        return True
        
    except Exception as e:
        print(f" Node testing failed: {e}")
        return False

if __name__ == "__main__":
    test_nodes()
