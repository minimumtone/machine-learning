from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chunking_strategies import Chunk
import numpy as np
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int

class RAGSystem:
    def __init__(self, collection_name: str = "rag_chunks"):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection_name = collection_name
        self.collection = None
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.chunks = []
        
    def initialize_collection(self, chunks: List[Chunk]):
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.chunks = chunks
        
        documents = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(documents, show_progress_bar=True)
        
        ids = [f"{chunk.method}_{chunk.chunk_id}" for chunk in chunks]
        metadatas = [
            {
                "method": chunk.method,
                "chunk_id": str(chunk.chunk_id),
                **{k: str(v) for k, v in chunk.metadata.items()}
            }
            for chunk in chunks
        ]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection first.")
        
        query_embedding = self.model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        retrieval_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            chunk_id = int(metadata['chunk_id'])
            method = metadata['method']
            
            chunk = next((c for c in self.chunks if c.chunk_id == chunk_id and c.method == method), None)
            
            if chunk:
                score = 1 - distance
                retrieval_results.append(RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=i + 1
                ))
        
        return retrieval_results
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.chunks:
            return {}
        
        method_stats = {}
        for method in set(chunk.method for chunk in self.chunks):
            method_chunks = [c for c in self.chunks if c.method == method]
            chunk_lengths = [len(c.content) for c in method_chunks]
            
            method_stats[method] = {
                'num_chunks': len(method_chunks),
                'avg_chunk_length': np.mean(chunk_lengths),
                'std_chunk_length': np.std(chunk_lengths),
                'min_chunk_length': np.min(chunk_lengths),
                'max_chunk_length': np.max(chunk_lengths),
                'median_chunk_length': np.median(chunk_lengths)
            }
        
        return method_stats
