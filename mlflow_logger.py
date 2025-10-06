"""
MLflow Logger module for Agentic RAG System
Handles comprehensive logging and observability
"""

import mlflow
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

def log_rag_run(query: str, retrieved_docs: List[Dict], initial_answer: str, 
                critique_result: str, refined_answer: str = None, 
                refinement_needed: bool = False, run_name: str = None) -> str:
    """
    Log complete RAG run to MLflow
    
    Args:
        query: Original user query
        retrieved_docs: Retrieved knowledge base documents
        initial_answer: Initial answer from LLM
        critique_result: Result from self-critique (COMPLETE/REFINE)
        refined_answer: Refined answer (if refinement was needed)
        refinement_needed: Whether refinement was required
        run_name: Optional name for the run
        
    Returns:
        MLflow run ID
    """
    
    try:
        # Create run name if not provided
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"agentic_rag_{timestamp}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_param("query", query)
            mlflow.log_param("retrieved_count", len(retrieved_docs))
            mlflow.log_param("refinement_needed", refinement_needed)
            mlflow.log_param("critique_result", critique_result)
            mlflow.log_param("pipeline_version", "1.0.0")
            
            # Log metrics
            doc_scores = [doc.get('score', 0) for doc in retrieved_docs]
            if doc_scores:
                mlflow.log_metric("avg_retrieval_score", sum(doc_scores) / len(doc_scores))
                mlflow.log_metric("max_retrieval_score", max(doc_scores))
                mlflow.log_metric("min_retrieval_score", min(doc_scores))
            
            mlflow.log_metric("initial_answer_length", len(initial_answer))
            if refined_answer:
                mlflow.log_metric("refined_answer_length", len(refined_answer))
                
            # Log number of citations in answers
            initial_citations = initial_answer.count("[KB")
            mlflow.log_metric("initial_answer_citations", initial_citations)
            if refined_answer:
                refined_citations = refined_answer.count("[KB")
                mlflow.log_metric("refined_answer_citations", refined_citations)
            
            # Log retrieved documents
            doc_summaries = []
            for doc in retrieved_docs:
                doc_summary = {
                    "doc_id": doc['id'],
                    "score": doc['score'],
                    "question": doc['metadata'].get('question', ''),
                    "source": doc['metadata'].get('source', ''),
                    "confidence": doc['metadata'].get('confidence_indicator', ''),
                    "last_updated": doc['metadata'].get('last_updated', '')
                }
                doc_summaries.append(doc_summary)
            
            mlflow.log_text(json.dumps(doc_summaries, indent=2), "retrieved_docs.json")
            
            # Log answers
            mlflow.log_text(initial_answer, "initial_answer.txt")
            if refined_answer:
                mlflow.log_text(refined_answer, "refined_answer.txt")
            
            # Log final result
            final_answer = refined_answer if refined_answer else initial_answer
            mlflow.log_text(final_answer, "final_answer.txt")
            
            # Log pipeline decision summary
            decision_summary = {
                "query": query,
                "retrieved_count": len(retrieved_docs),
                "critique_result": critique_result,
                "refinement_needed": refinement_needed,
                "final_answer_length": len(final_answer),
                "timestamp": datetime.now().isoformat()
            }
            mlflow.log_text(json.dumps(decision_summary, indent=2), "decision_summary.json")
            
            print(f"✅ MLflow run logged: {run.info.run_id}")
            print(f"   Run name: {run_name}")
            print(f"   Retrieved docs: {len(retrieved_docs)}")
            print(f"   Refinement needed: {refinement_needed}")
            print(f"   Final answer length: {len(final_answer)} chars")
            
            return run.info.run_id
            
    except Exception as e:
        print(f"❌ Failed to log to MLflow: {e}")
        return None

def log_experiment_summary(total_queries: int, successful_runs: int, 
                          refinement_rate: float, avg_retrieval_score: float) -> str:
    """
    Log summary statistics for an entire experiment
    
    Args:
        total_queries: Total number of queries processed
        successful_runs: Number of successful runs
        refinement_rate: Percentage of queries that required refinement
        avg_retrieval_score: Average retrieval score across all queries
        
    Returns:
        Experiment run ID
    """
    
    try:
        with mlflow.start_run(run_name="experiment_summary") as run:
            mlflow.log_param("experiment_type", "agentic_rag_evaluation")
            mlflow.log_param("total_queries", total_queries)
            mlflow.log_param("successful_runs", successful_runs)
            
            mlflow.log_metric("success_rate", (successful_runs / total_queries) * 100)
            mlflow.log_metric("refinement_rate", refinement_rate)
            mlflow.log_metric("avg_retrieval_score", avg_retrieval_score)
            
            summary_data = {
                "total_queries": total_queries,
                "successful_runs": successful_runs,
                "success_rate": (successful_runs / total_queries) * 100,
                "refinement_rate": refinement_rate,
                "avg_retrieval_score": avg_retrieval_score,
                "timestamp": datetime.now().isoformat()
            }
            
            mlflow.log_text(json.dumps(summary_data, indent=2), "experiment_summary.json")
            
            print(f"✅ Experiment summary logged: {run.info.run_id}")
            return run.info.run_id
            
    except Exception as e:
        print(f"❌ Failed to log experiment summary: {e}")
        return None

def list_recent_runs(limit: int = 10) -> List[Dict[str, Any]]:
    """
    List recent MLflow runs
    
    Args:
        limit: Maximum number of runs to return
        
    Returns:
        List of run information
    """
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        runs = client.search_runs(order_by=["start_time desc"], max_results=limit)
        
        run_info = []
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", "unnamed"),
                "start_time": run.info.start_time,
                "status": run.info.status,
                "query": run.data.params.get("query", "unknown")
            }
            run_info.append(run_data)
        
        print(f"✅ Retrieved {len(run_info)} recent runs")
        return run_info
        
    except Exception as e:
        print(f"❌ Failed to list recent runs: {e}")
        return []

def test_mlflow_logging():
    """Test MLflow logging functionality"""
    print("\n🧪 Testing MLflow Logging...")
    
    # Test basic logging
    test_data = {
        "query": "What are best practices for caching?",
        "retrieved_docs": [
            {
                "id": "KB001",
                "score": 0.85,
                "metadata": {
                    "question": "What are best practices for caching?",
                    "source": "caching_guide.md",
                    "confidence_indicator": "high",
                    "last_updated": "2024-01-10"
                }
            }
        ],
        "initial_answer": "Caching best practices include implementing proper cache invalidation, using appropriate cache sizes, and monitoring cache hit rates. [KB001]",
        "critique_result": "COMPLETE",
        "refinement_needed": False,
        "refined_answer": None
    }
    
    try:
        run_id = log_rag_run(
            query=test_data["query"],
            retrieved_docs=test_data["retrieved_docs"],
            initial_answer=test_data["initial_answer"],
            critique_result=test_data["critique_result"],
            refinement_needed=test_data["refinement_needed"],
            run_name="test_run"
        )
        
        if run_id:
            print("✅ MLflow logging test successful!")
            
            # Test experiment summary
            summary_run_id = log_experiment_summary(
                total_queries=5,
                successful_runs=5,
                refinement_rate=20.0,
                avg_retrieval_score=0.78
            )
            
            if summary_run_id:
                print("✅ Experiment summary logging successful!")
            
            return True
        else:
            print("❌ MLflow logging test failed")
            return False
            
    except Exception as e:
        print(f"❌ MLflow logging test error: {e}")
        return False

if __name__ == "__main__":
    test_mlflow_logging()
