import sys
from pathlib import Path

from pdf_to_markdown import PDFToMarkdownConverter
from chunking_strategies import MarkdownChunker, MeCabSemanticChunker, SelfRouteChunker
from rag_system import RAGSystem
from evaluation import ChunkingEvaluator

def test_pdf_conversion():
    print("=" * 80)
    print("Testing PDF to Markdown Conversion")
    print("=" * 80)
    
    pdf_path = "/home/ubuntu/attachments/30af5be2-cb9a-4976-9c52-fcbb02d7b303/CRDS-FY2024-FR-09.pdf"
    
    converter = PDFToMarkdownConverter(pdf_path)
    info = converter.get_pdf_info()
    
    print(f"PDF Pages: {info['num_pages']}")
    print(f"File Size: {info['file_size_mb']:.2f} MB")
    
    md_text = converter.convert_to_markdown()
    print(f"Markdown Length: {len(md_text):,} characters")
    print(f"First 500 characters:\n{md_text[:500]}")
    
    return md_text

def test_chunking_methods(md_text):
    print("\n" + "=" * 80)
    print("Testing Chunking Methods")
    print("=" * 80)
    
    chunk_size = 1000
    chunk_overlap = 200
    similarity_threshold = 0.7
    
    print("\nParameters:")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Chunk Overlap: {chunk_overlap}")
    print(f"  Similarity Threshold: {similarity_threshold}")
    
    sample_text = md_text[:50000]
    
    print("\n--- Markdown-based Chunking ---")
    markdown_chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    markdown_chunks = markdown_chunker.chunk(sample_text)
    print(f"Number of chunks: {len(markdown_chunks)}")
    if markdown_chunks:
        print(f"First chunk length: {len(markdown_chunks[0].content)}")
        print(f"First chunk preview: {markdown_chunks[0].content[:200]}...")
    
    print("\n--- MeCab Semantic Chunking ---")
    mecab_chunker = MeCabSemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    mecab_chunks = mecab_chunker.chunk(sample_text)
    print(f"Number of chunks: {len(mecab_chunks)}")
    if mecab_chunks:
        print(f"First chunk length: {len(mecab_chunks[0].content)}")
        print(f"First chunk preview: {mecab_chunks[0].content[:200]}...")
    
    print("\n--- Self-Route Chunking ---")
    self_route_chunker = SelfRouteChunker(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        similarity_threshold=similarity_threshold
    )
    self_route_chunks = self_route_chunker.chunk(sample_text)
    print(f"Number of chunks: {len(self_route_chunks)}")
    if self_route_chunks:
        print(f"First chunk length: {len(self_route_chunks[0].content)}")
        print(f"First chunk preview: {self_route_chunks[0].content[:200]}...")
    
    return {
        'markdown': markdown_chunks,
        'mecab_semantic': mecab_chunks,
        'self_route': self_route_chunks
    }

def test_evaluation(chunks_by_method):
    print("\n" + "=" * 80)
    print("Testing Evaluation Metrics")
    print("=" * 80)
    
    evaluator = ChunkingEvaluator()
    comparison = evaluator.compare_methods(chunks_by_method)
    
    for method, metrics in comparison.items():
        print(f"\n--- {method} ---")
        print(f"  Number of chunks: {metrics['num_chunks']}")
        print(f"  Average chunk length: {metrics['avg_chunk_length']:.2f}")
        print(f"  Median chunk length: {metrics['median_chunk_length']:.2f}")
        print(f"  Std chunk length: {metrics['std_chunk_length']:.2f}")
        print(f"  Min chunk length: {metrics['min_chunk_length']}")
        print(f"  Max chunk length: {metrics['max_chunk_length']}")
        print(f"  Length variance: {metrics['length_variance']:.2f}")
        print(f"  Avg inter-chunk coherence: {metrics['avg_inter_chunk_coherence']:.3f}")

def test_rag_system(chunks_by_method):
    print("\n" + "=" * 80)
    print("Testing RAG System")
    print("=" * 80)
    
    test_queries = [
        "科学技術政策の動向について教えてください",
        "イノベーションに関する情報",
        "研究開発の俯瞰"
    ]
    
    evaluator = ChunkingEvaluator()
    
    for method, chunks in chunks_by_method.items():
        print(f"\n--- Testing {method} ---")
        
        rag_system = RAGSystem(collection_name=f"test_{method}")
        num_chunks = rag_system.initialize_collection(chunks)
        print(f"Initialized collection with {num_chunks} chunks")
        
        for query in test_queries[:1]:
            print(f"\nQuery: {query}")
            results = rag_system.retrieve(query, top_k=3)
            
            print(f"Retrieved {len(results)} results")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: Score={result.score:.3f}, Length={len(result.chunk.content)}")
            
            eval_metrics = evaluator.evaluate_retrieval_quality(query, results)
            print(f"  Avg Score: {eval_metrics['avg_score']:.3f}")
            print(f"  Avg Relevance: {eval_metrics['avg_relevance']:.3f}")

def main():
    print("Starting Comprehensive Testing of RAG Chunking Comparison System")
    print("=" * 80)
    
    try:
        md_text = test_pdf_conversion()
        
        chunks_by_method = test_chunking_methods(md_text)
        
        test_evaluation(chunks_by_method)
        
        test_rag_system(chunks_by_method)
        
        print("\n" + "=" * 80)
        print("All Tests Completed Successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
