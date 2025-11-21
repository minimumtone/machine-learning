from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from paraphrase_generator import AugmentedChunk
from augmented_rag_system import AugmentedRetrievalResult

class AugmentedChunkingEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def evaluate_paraphrase_quality(self, augmented_chunks: List[AugmentedChunk]) -> Dict[str, Any]:
        all_similarities = []
        all_diversities = []
        
        for aug_chunk in augmented_chunks:
            if len(aug_chunk.all_variations) < 2:
                continue
            
            embeddings = self.model.encode(aug_chunk.all_variations)
            
            original_embedding = embeddings[0]
            paraphrase_embeddings = embeddings[1:]
            
            similarities = []
            for para_emb in paraphrase_embeddings:
                similarity = np.dot(original_embedding, para_emb) / (
                    np.linalg.norm(original_embedding) * np.linalg.norm(para_emb)
                )
                similarities.append(similarity)
            
            all_similarities.extend(similarities)
            
            if len(paraphrase_embeddings) > 1:
                diversity_scores = []
                for i in range(len(paraphrase_embeddings)):
                    for j in range(i + 1, len(paraphrase_embeddings)):
                        diversity = 1 - (np.dot(paraphrase_embeddings[i], paraphrase_embeddings[j]) / (
                            np.linalg.norm(paraphrase_embeddings[i]) * np.linalg.norm(paraphrase_embeddings[j])
                        ))
                        diversity_scores.append(diversity)
                
                if diversity_scores:
                    all_diversities.extend(diversity_scores)
        
        return {
            'avg_similarity_to_original': np.mean(all_similarities) if all_similarities else 0,
            'std_similarity_to_original': np.std(all_similarities) if all_similarities else 0,
            'min_similarity': np.min(all_similarities) if all_similarities else 0,
            'max_similarity': np.max(all_similarities) if all_similarities else 0,
            'avg_paraphrase_diversity': np.mean(all_diversities) if all_diversities else 0,
            'std_paraphrase_diversity': np.std(all_diversities) if all_diversities else 0,
            'num_evaluated_chunks': len(augmented_chunks),
            'total_paraphrases': len(all_similarities)
        }
    
    def evaluate_retrieval_improvement(
        self,
        queries: List[str],
        augmented_results_list: List[List[AugmentedRetrievalResult]],
        baseline_results_list: List[List[Any]]
    ) -> Dict[str, Any]:
        if len(queries) != len(augmented_results_list) or len(queries) != len(baseline_results_list):
            raise ValueError("Number of queries must match number of result lists")
        
        score_improvements = []
        variation_usage_counts = {'original': 0, 'paraphrase': 0}
        overlap_ratios = []
        
        for aug_results, base_results in zip(augmented_results_list, baseline_results_list):
            if aug_results and base_results:
                aug_avg_score = np.mean([r.score for r in aug_results])
                base_avg_score = np.mean([r.score for r in base_results])
                score_improvements.append(aug_avg_score - base_avg_score)
                
                for r in aug_results:
                    if r.variation_index == 0:
                        variation_usage_counts['original'] += 1
                    else:
                        variation_usage_counts['paraphrase'] += 1
                
                aug_chunk_ids = set((r.original_chunk.method, r.original_chunk.chunk_id) for r in aug_results)
                base_chunk_ids = set((r.chunk.method, r.chunk.chunk_id) for r in base_results)
                overlap = len(aug_chunk_ids & base_chunk_ids)
                overlap_ratios.append(overlap / len(aug_results) if aug_results else 0)
        
        total_variations = sum(variation_usage_counts.values())
        paraphrase_usage_ratio = variation_usage_counts['paraphrase'] / total_variations if total_variations > 0 else 0
        
        return {
            'avg_score_improvement': np.mean(score_improvements) if score_improvements else 0,
            'std_score_improvement': np.std(score_improvements) if score_improvements else 0,
            'positive_improvements': sum(1 for x in score_improvements if x > 0),
            'negative_improvements': sum(1 for x in score_improvements if x < 0),
            'paraphrase_usage_ratio': paraphrase_usage_ratio,
            'variation_usage_counts': variation_usage_counts,
            'avg_overlap_ratio': np.mean(overlap_ratios) if overlap_ratios else 0,
            'num_queries_evaluated': len(queries)
        }
    
    def evaluate_augmented_chunks_by_method(self, augmented_chunks: List[AugmentedChunk]) -> Dict[str, Dict[str, Any]]:
        method_stats = {}
        
        for method in set(ac.original_chunk.method for ac in augmented_chunks):
            method_chunks = [ac for ac in augmented_chunks if ac.original_chunk.method == method]
            
            total_variations = sum(len(ac.all_variations) for ac in method_chunks)
            avg_variations = total_variations / len(method_chunks) if method_chunks else 0
            
            original_lengths = [len(ac.original_chunk.content) for ac in method_chunks]
            
            all_paraphrase_lengths = []
            for ac in method_chunks:
                all_paraphrase_lengths.extend([len(p) for p in ac.paraphrases])
            
            method_stats[method] = {
                'num_chunks': len(method_chunks),
                'total_variations': total_variations,
                'avg_variations_per_chunk': avg_variations,
                'avg_original_length': np.mean(original_lengths) if original_lengths else 0,
                'avg_paraphrase_length': np.mean(all_paraphrase_lengths) if all_paraphrase_lengths else 0,
                'length_ratio': (np.mean(all_paraphrase_lengths) / np.mean(original_lengths)) if original_lengths and all_paraphrase_lengths else 0
            }
        
        return method_stats
    
    def generate_comprehensive_report(
        self,
        augmented_chunks: List[AugmentedChunk],
        queries: List[str],
        augmented_results_list: List[List[AugmentedRetrievalResult]],
        baseline_results_list: List[List[Any]]
    ) -> Dict[str, Any]:
        paraphrase_quality = self.evaluate_paraphrase_quality(augmented_chunks)
        retrieval_improvement = self.evaluate_retrieval_improvement(queries, augmented_results_list, baseline_results_list)
        method_stats = self.evaluate_augmented_chunks_by_method(augmented_chunks)
        
        return {
            'paraphrase_quality': paraphrase_quality,
            'retrieval_improvement': retrieval_improvement,
            'method_statistics': method_stats,
            'summary': {
                'total_original_chunks': len(augmented_chunks),
                'total_variations': sum(len(ac.all_variations) for ac in augmented_chunks),
                'avg_variations_per_chunk': np.mean([len(ac.all_variations) for ac in augmented_chunks]) if augmented_chunks else 0,
                'paraphrase_quality_score': paraphrase_quality.get('avg_similarity_to_original', 0),
                'retrieval_improvement_score': retrieval_improvement.get('avg_score_improvement', 0),
                'paraphrase_usage_ratio': retrieval_improvement.get('paraphrase_usage_ratio', 0)
            }
        }
