from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chunking_strategies import Chunk
from paraphrase_generator import AugmentedChunk, augment_chunks
import numpy as np
from dataclasses import dataclass

@dataclass
class AugmentedRetrievalResult:
    original_chunk: Chunk
    matched_variation: str
    variation_index: int
    score: float
    rank: int

class AugmentedRAGSystem:
    def __init__(self, collection_name: str = "augmented_rag_chunks"):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection_name = collection_name
        self.collection = None
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.augmented_chunks = []
        
    def initialize_collection(self, augmented_chunks: List[AugmentedChunk]):
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.augmented_chunks = augmented_chunks
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for aug_chunk in augmented_chunks:
            original_chunk = aug_chunk.original_chunk
            
            for var_idx, variation in enumerate(aug_chunk.all_variations):
                all_documents.append(variation)
                
                metadata = {
                    "method": original_chunk.method,
                    "original_chunk_id": str(original_chunk.chunk_id),
                    "variation_index": str(var_idx),
                    "is_original": str(var_idx == 0),
                    **{k: str(v) for k, v in original_chunk.metadata.items()}
                }
                all_metadatas.append(metadata)
                
                chunk_id = f"{original_chunk.method}_{original_chunk.chunk_id}_var{var_idx}"
                all_ids.append(chunk_id)
        
        print(f"Encoding {len(all_documents)} document variations...")
        embeddings = self.model.encode(all_documents, show_progress_bar=True, batch_size=32)
        
        print(f"Adding {len(all_documents)} variations to collection...")
        batch_size = 1000
        for i in range(0, len(all_documents), batch_size):
            end_idx = min(i + batch_size, len(all_documents))
            self.collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=all_documents[i:end_idx],
                metadatas=all_metadatas[i:end_idx],
                ids=all_ids[i:end_idx]
            )
        
        total_variations = len(all_documents)
        original_chunks_count = len(augmented_chunks)
        avg_variations = total_variations / original_chunks_count if original_chunks_count > 0 else 0
        
        print(f"Collection initialized with {total_variations} variations from {original_chunks_count} original chunks")
        print(f"Average variations per chunk: {avg_variations:.1f}")
        
        return total_variations
    
    def retrieve(self, query: str, top_k: int = 5, return_unique_chunks: bool = True) -> List[AugmentedRetrievalResult]:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection first.")
        
        query_embedding = self.model.encode([query])[0]
        
        search_k = top_k * 3 if return_unique_chunks else top_k
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=search_k
        )
        
        retrieval_results = []
        seen_chunk_ids = set()
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            original_chunk_id = int(metadata['original_chunk_id'])
            method = metadata['method']
            variation_index = int(metadata['variation_index'])
            
            if return_unique_chunks and (method, original_chunk_id) in seen_chunk_ids:
                continue
            
            aug_chunk = next(
                (ac for ac in self.augmented_chunks 
                 if ac.original_chunk.chunk_id == original_chunk_id 
                 and ac.original_chunk.method == method), 
                None
            )
            
            if aug_chunk:
                score = 1 - distance
                retrieval_results.append(AugmentedRetrievalResult(
                    original_chunk=aug_chunk.original_chunk,
                    matched_variation=doc,
                    variation_index=variation_index,
                    score=score,
                    rank=len(retrieval_results) + 1
                ))
                
                if return_unique_chunks:
                    seen_chunk_ids.add((method, original_chunk_id))
                
                if len(retrieval_results) >= top_k:
                    break
        
        return retrieval_results
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.augmented_chunks:
            return {}
        
        method_stats = {}
        for method in set(ac.original_chunk.method for ac in self.augmented_chunks):
            method_chunks = [ac for ac in self.augmented_chunks if ac.original_chunk.method == method]
            
            total_variations = sum(len(ac.all_variations) for ac in method_chunks)
            avg_variations = total_variations / len(method_chunks) if method_chunks else 0
            
            chunk_lengths = [len(ac.original_chunk.content) for ac in method_chunks]
            
            method_stats[method] = {
                'num_original_chunks': len(method_chunks),
                'total_variations': total_variations,
                'avg_variations_per_chunk': avg_variations,
                'avg_original_chunk_length': np.mean(chunk_lengths),
                'std_original_chunk_length': np.std(chunk_lengths),
                'min_chunk_length': np.min(chunk_lengths),
                'max_chunk_length': np.max(chunk_lengths)
            }
        
        return method_stats
    
    def compare_with_baseline(self, query: str, baseline_results: List, top_k: int = 5) -> Dict[str, Any]:
        augmented_results = self.retrieve(query, top_k=top_k)
        
        augmented_chunk_ids = set(
            (r.original_chunk.method, r.original_chunk.chunk_id) 
            for r in augmented_results
        )
        baseline_chunk_ids = set(
            (r.chunk.method, r.chunk.chunk_id) 
            for r in baseline_results
        )
        
        overlap = len(augmented_chunk_ids & baseline_chunk_ids)
        overlap_ratio = overlap / top_k if top_k > 0 else 0
        
        augmented_avg_score = np.mean([r.score for r in augmented_results]) if augmented_results else 0
        baseline_avg_score = np.mean([r.score for r in baseline_results]) if baseline_results else 0
        
        variation_usage = {}
        for r in augmented_results:
            var_type = "original" if r.variation_index == 0 else f"variation_{r.variation_index}"
            variation_usage[var_type] = variation_usage.get(var_type, 0) + 1
        
        return {
            'overlap_count': overlap,
            'overlap_ratio': overlap_ratio,
            'augmented_avg_score': augmented_avg_score,
            'baseline_avg_score': baseline_avg_score,
            'score_improvement': augmented_avg_score - baseline_avg_score,
            'variation_usage': variation_usage,
            'unique_augmented_chunks': len(augmented_chunk_ids),
            'unique_baseline_chunks': len(baseline_chunk_ids)
        }
