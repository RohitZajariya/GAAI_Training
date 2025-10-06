"""
Main execution module for Agentic RAG System
Conducts comprehensive integration testing and sample queries
"""

from data_loader import load_kb_data, test_data_loader
from clients import initialize_system, client_manager, test_clients
from rag_nodes import run_agentic_rag_pipeline, test_nodes
from mlflow_logger import log_rag_run, test_mlflow_logging
from config import validate_config

# Sample queries from assignment
SAMPLE_QUERIES = [
    "What are best practices for caching?",
    "How should I set up CI/CD pipelines?", 
    "What are performance tuning tips?",
    "How do I version my APIs?",
    "What should I consider for error handling?"
]

def run_sample_queries() -> list:
    """
    Run all sample queries through the agentic RAG system
    
    Returns:
        List of results from each query
    """
    
    print("🧪 Testing Agentic RAG System with Sample Queries...")
    print("=" * 70)
    
    results = []
    
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n📋 Test {i}/{len(SAMPLE_QUERIES)}")
        print(f"Query: {query}")
        
        # Run the agentic RAG pipeline
        result = run_agentic_rag_pipeline(query)
        results.append(result)
        
        # Log to MLflow
        if 'error' not in result:
            mlflow_run_id = log_rag_run(
                query=result['query'],
                retrieved_docs=result['retrieved_docs'],
                initial_answer=result['initial_answer'],
                critique_result=result['critique_result'],
                refinement_needed=result['refinement_needed'],
                refined_answer=result.get('refined_answer')
            )
            result['mlflow_run_id'] = mlflow_run_id
        
        # Display results
        print(f"\n📝 Result Summary:")
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"Retrieved docs: {len(result['retrieved_docs'])}")
            print(f"Critique result: {result['critique_result']}")
            print(f"Refinement needed: {result['refinement_needed']}")
            print(f"Final answer preview: {result['final_answer'][:200]}...")
            print(f"MLflow Run ID: {result.get('mlflow_run_id', 'N/A')}")
        
        if i < len(SAMPLE_QUERIES):
            print("\n" + "-" * 70)
    
    print(f"\n🎯 Testing completed! Processed {len(results)} queries")
    return results

def analyze_results(results: list):
    """
    Analyze the results from sample queries
    
    Args:
        results: List of query results
    """
    
    print("\n📊 Results Analysis:")
    print("=" * 50)
    
    total_queries = len(results)
    successful_queries = sum(1 for r in results if 'error' not in r)
    refinement_count = sum(1 for r in results if r.get('refinement_needed', False))
    
    print(f"Total queries: {total_queries}")
    print(f"Successful queries: {successful_queries}")
    print(f"Success rate: {(successful_queries/total_queries)*100:.1f}%")
    print(f"Refinement rate: {(refinement_count/total_queries)*100:.1f}%")
    
    # Calculate average retrieval scores
    avg_scores = []
    citation_counts = []
    
    for result in results:
        if 'error' not in result:
            # Average retrieval score
            scores = [doc['score'] for doc in result['retrieved_docs']]
            if scores:
                avg_scores.append(sum(scores) / len(scores))
            
            # Citation count
            citations = result['final_answer'].count('[KB')
            citation_counts.append(citations)
    
    if avg_scores:
        print(f"Average retrieval score: {sum(avg_scores)/len(avg_scores):.3f}")
    
    if citation_counts:
        print(f"Average citations per answer: {sum(citation_counts)/len(citation_counts):.1f}")
    
    # Log experiment summary to MLflow
    if avg_scores:
        avg_retrieval_score = sum(avg_scores) / len(avg_scores)
        refinement_rate = (refinement_count / total_queries) * 100
        
        from mlflow_logger import log_experiment_summary
        log_experiment_summary(
            total_queries=total_queries,
            successful_runs=successful_queries,
            refinement_rate=refinement_rate,
            avg_retrieval_score=avg_retrieval_score
        )
    
    return {
        'total_queries': total_queries,
        'successful_queries': successful_queries,
        'success_rate': (successful_queries/total_queries)*100,
        'refinement_rate': (refinement_count/total_queries)*100,
        'avg_retrieval_score': sum(avg_scores)/len(avg_scores) if avg_scores else 0
    }

def comprehensive_test():
    """
    Run comprehensive system test
    """
    
    print("🚀 Agentic RAG System - Comprehensive Test")
    print("=" * 60)
    
    # Step 1: Validate configuration
    print("\n1️⃣ Validating Configuration...")
    if not validate_config():
        print("❌ Configuration validation failed. Please update API keys.")
        return False
    
    # Step 2: Test data loader
    print("\n2️⃣ Testing Data Loader...")
    if not test_data_loader():
        print("❌ Data loader test failed")
        return False
    
    # Step 3: Load KB data
    print("\n3️⃣ Loading Knowledge Base...")
    kb_data = load_kb_data()
    if not kb_data:
        print("❌ Failed to load knowledge base")
        return False
    
    # Step 4: Initialize system
    print("\n4️⃣ Initializing System...")
    if not initialize_system(kb_data):
        print("❌ System initialization failed")
        return False
    
    # Step 5: Test clients
    print("\n5️⃣ Testing Clients...")
    if not test_clients():
        print("❌ Client testing failed")
        return False
    
    # Step 6: Test nodes
    print("\n6️⃣ Testing RAG Nodes...")
    if not test_nodes():
        print("❌ Node testing failed")
        return False
    
    # Step 7: Test MLflow logging
    print("\n7️⃣ Testing MLflow Logging...")
    if not test_mlflow_logging():
        print("❌ MLflow logging test failed")
        return False
    
    # Step 8: Run sample queries
    print("\n8️⃣ Running Sample Queries...")
    results = run_sample_queries()
    
    # Step 9: Analyze results
    print("\n9️⃣ Analyzing Results...")
    analysis = analyze_results(results)
    
    # Final summary
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"System Status: {'✅ FULLY FUNCTIONAL' if analysis['success_rate'] >= 80 else '⚠️ NEEDS ATTENTION'}")
    print(f"Success Rate: {analysis['success_rate']:.1f}%")
    print(f"Refinement Rate: {analysis['refinement_rate']:.1f}%")
    print(f"Average Retrieval Score: {analysis['avg_retrieval_score']:.3f}")
    
    return True

if __name__ == "__main__":
    comprehensive_test()
