from typing import List, Dict, Any
from chunking_strategies import Chunk
from rag_system import RetrievalResult
import numpy as np
from sentence_transformers import SentenceTransformer

class ChunkingEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def evaluate_chunk_quality(self, chunks: List[Chunk]) -> Dict[str, Any]:
        if not chunks:
            return {}
        
        chunk_lengths = [len(c.content) for c in chunks]
        
        coherence_scores = []
        for i in range(len(chunks) - 1):
            if chunks[i].method == chunks[i+1].method:
                embeddings = self.model.encode([chunks[i].content, chunks[i+1].content])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                coherence_scores.append(float(similarity))
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        variance = np.var(chunk_lengths)
        
        return {
            'num_chunks': len(chunks),
            'avg_chunk_length': float(np.mean(chunk_lengths)),
            'std_chunk_length': float(np.std(chunk_lengths)),
            'min_chunk_length': int(np.min(chunk_lengths)),
            'max_chunk_length': int(np.max(chunk_lengths)),
            'median_chunk_length': float(np.median(chunk_lengths)),
            'length_variance': float(variance),
            'avg_inter_chunk_coherence': float(avg_coherence),
            'coherence_samples': len(coherence_scores)
        }
    
    def evaluate_retrieval_quality(
        self, 
        query: str, 
        retrieval_results: List[RetrievalResult],
        ground_truth_keywords: List[str] = None
    ) -> Dict[str, Any]:
        if not retrieval_results:
            return {}
        
        scores = [r.score for r in retrieval_results]
        
        query_embedding = self.model.encode([query])[0]
        chunk_embeddings = self.model.encode([r.chunk.content for r in retrieval_results])
        
        relevance_scores = []
        for emb in chunk_embeddings:
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            relevance_scores.append(float(similarity))
        
        keyword_coverage = 0.0
        if ground_truth_keywords:
            total_keywords = len(ground_truth_keywords)
            found_keywords = 0
            for keyword in ground_truth_keywords:
                for result in retrieval_results:
                    if keyword.lower() in result.chunk.content.lower():
                        found_keywords += 1
                        break
            keyword_coverage = found_keywords / total_keywords if total_keywords > 0 else 0.0
        
        return {
            'num_results': len(retrieval_results),
            'avg_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'avg_relevance': float(np.mean(relevance_scores)),
            'keyword_coverage': float(keyword_coverage),
            'top_1_score': float(scores[0]) if scores else 0.0,
            'top_3_avg_score': float(np.mean(scores[:3])) if len(scores) >= 3 else float(np.mean(scores))
        }
    
    def compare_methods(
        self, 
        chunks_by_method: Dict[str, List[Chunk]]
    ) -> Dict[str, Dict[str, Any]]:
        comparison = {}
        
        for method, chunks in chunks_by_method.items():
            comparison[method] = self.evaluate_chunk_quality(chunks)
        
        return comparison
