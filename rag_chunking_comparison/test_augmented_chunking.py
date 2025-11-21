import os
import sys
from pathlib import Path

from pdf_to_markdown import convert_pdf_to_markdown, get_pdf_info
from chunking_strategies import MarkdownChunker, MeCabSemanticChunker, SelfRouteChunker
from paraphrase_generator import augment_chunks
from augmented_rag_system import AugmentedRAGSystem
from augmented_evaluation import AugmentedChunkingEvaluator
from rag_system import RAGSystem

def test_paraphrase_generation():
    print("=" * 80)
    print("TEST 1: Paraphrase Generation")
    print("=" * 80)
    
    from paraphrase_generator import ParaphraseGenerator
    
    sample_text = "科学技術政策の動向について、主要国・地域の取り組みを分析し、日本の政策立案に活用することが重要です。"
    
    print(f"\nOriginal text:\n{sample_text}\n")
    
    try:
        generator = ParaphraseGenerator()
        paraphrases = generator.generate_paraphrases(sample_text, num_variations=5)
        
        print(f"Generated {len(paraphrases)} paraphrases:")
        for i, p in enumerate(paraphrases, 1):
            print(f"{i}. {p}")
        
        print("\n✓ Paraphrase generation test passed!")
        return True
    except Exception as e:
        print(f"\n✗ Paraphrase generation test failed: {e}")
        return False

def test_augmented_chunking_and_rag(pdf_path: str, num_chunks_to_test: int = 3):
    print("\n" + "=" * 80)
    print("TEST 2: Augmented Chunking and RAG System")
    print("=" * 80)
    
    if not os.path.exists(pdf_path):
        print(f"\n✗ PDF file not found: {pdf_path}")
        return False
    
    print(f"\nPDF: {pdf_path}")
    pdf_info = get_pdf_info(pdf_path)
    print(f"Pages: {pdf_info['num_pages']}, Size: {pdf_info['file_size_mb']:.2f} MB")
    
    print("\nConverting PDF to markdown...")
    markdown_text = convert_pdf_to_markdown(pdf_path)
    print(f"Converted to {len(markdown_text)} characters")
    
    print("\nCreating chunks with MeCab Semantic method...")
    chunker = MeCabSemanticChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk(markdown_text)
    print(f"Created {len(chunks)} chunks")
    
    chunks_to_augment = chunks[:num_chunks_to_test]
    print(f"\nAugmenting first {num_chunks_to_test} chunks with 3 paraphrases each...")
    
    try:
        augmented_chunks = augment_chunks(chunks_to_augment, num_variations=3)
        
        print(f"\nAugmented {len(augmented_chunks)} chunks:")
        for i, aug_chunk in enumerate(augmented_chunks):
            print(f"\nChunk {i+1} (ID: {aug_chunk.original_chunk.chunk_id}):")
            print(f"  Original length: {len(aug_chunk.original_chunk.content)} chars")
            print(f"  Paraphrases: {len(aug_chunk.paraphrases)}")
            print(f"  Total variations: {len(aug_chunk.all_variations)}")
            
            if aug_chunk.paraphrases:
                print(f"  Sample paraphrase: {aug_chunk.paraphrases[0][:100]}...")
        
        print("\nInitializing Augmented RAG System...")
        aug_rag = AugmentedRAGSystem(collection_name="test_augmented_rag")
        total_variations = aug_rag.initialize_collection(augmented_chunks)
        print(f"Initialized with {total_variations} total variations")
        
        print("\nInitializing Baseline RAG System for comparison...")
        baseline_rag = RAGSystem(collection_name="test_baseline_rag")
        baseline_rag.initialize_collection(chunks_to_augment)
        
        test_queries = [
            "科学技術政策の動向について教えてください",
            "研究開発の取り組みについて",
            "技術革新の現状"
        ]
        
        print("\nTesting retrieval with sample queries:")
        for query in test_queries:
            print(f"\n  Query: {query}")
            
            aug_results = aug_rag.retrieve(query, top_k=3)
            baseline_results = baseline_rag.retrieve(query, top_k=3)
            
            print(f"  Augmented results: {len(aug_results)} chunks")
            for j, result in enumerate(aug_results[:2], 1):
                var_type = "original" if result.variation_index == 0 else f"variation {result.variation_index}"
                print(f"    {j}. Score: {result.score:.4f}, Matched: {var_type}")
                print(f"       Content: {result.original_chunk.content[:80]}...")
            
            print(f"  Baseline results: {len(baseline_results)} chunks")
            for j, result in enumerate(baseline_results[:2], 1):
                print(f"    {j}. Score: {result.score:.4f}")
                print(f"       Content: {result.chunk.content[:80]}...")
        
        print("\nEvaluating paraphrase quality...")
        evaluator = AugmentedChunkingEvaluator()
        quality_metrics = evaluator.evaluate_paraphrase_quality(augmented_chunks)
        
        print(f"  Avg similarity to original: {quality_metrics['avg_similarity_to_original']:.4f}")
        print(f"  Avg paraphrase diversity: {quality_metrics['avg_paraphrase_diversity']:.4f}")
        print(f"  Total paraphrases evaluated: {quality_metrics['total_paraphrases']}")
        
        print("\n✓ Augmented chunking and RAG test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Augmented chunking and RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_with_baseline(pdf_path: str, num_chunks_to_test: int = 3):
    print("\n" + "=" * 80)
    print("TEST 3: Comparison with Baseline RAG")
    print("=" * 80)
    
    if not os.path.exists(pdf_path):
        print(f"\n✗ PDF file not found: {pdf_path}")
        return False
    
    print("\nConverting PDF to markdown...")
    markdown_text = convert_pdf_to_markdown(pdf_path)
    
    print("\nCreating chunks...")
    chunker = MeCabSemanticChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk(markdown_text)
    chunks_to_test = chunks[:num_chunks_to_test]
    
    print(f"\nAugmenting {num_chunks_to_test} chunks with 3 paraphrases each...")
    
    try:
        augmented_chunks = augment_chunks(chunks_to_test, num_variations=3)
        
        print("\nInitializing systems...")
        aug_rag = AugmentedRAGSystem(collection_name="test_comparison_aug")
        aug_rag.initialize_collection(augmented_chunks)
        
        baseline_rag = RAGSystem(collection_name="test_comparison_base")
        baseline_rag.initialize_collection(chunks_to_test)
        
        test_queries = [
            "科学技術政策について",
            "研究開発の動向"
        ]
        
        print("\nComparing retrieval results:")
        all_aug_results = []
        all_baseline_results = []
        
        for query in test_queries:
            print(f"\n  Query: {query}")
            
            aug_results = aug_rag.retrieve(query, top_k=3)
            baseline_results = baseline_rag.retrieve(query, top_k=3)
            
            all_aug_results.append(aug_results)
            all_baseline_results.append(baseline_results)
            
            aug_avg_score = sum(r.score for r in aug_results) / len(aug_results) if aug_results else 0
            base_avg_score = sum(r.score for r in baseline_results) / len(baseline_results) if baseline_results else 0
            
            print(f"    Augmented avg score: {aug_avg_score:.4f}")
            print(f"    Baseline avg score: {base_avg_score:.4f}")
            print(f"    Improvement: {(aug_avg_score - base_avg_score):.4f}")
        
        print("\nEvaluating retrieval improvement...")
        evaluator = AugmentedChunkingEvaluator()
        improvement_metrics = evaluator.evaluate_retrieval_improvement(
            test_queries, all_aug_results, all_baseline_results
        )
        
        print(f"  Avg score improvement: {improvement_metrics['avg_score_improvement']:.4f}")
        print(f"  Paraphrase usage ratio: {improvement_metrics['paraphrase_usage_ratio']:.2%}")
        print(f"  Positive improvements: {improvement_metrics['positive_improvements']}/{improvement_metrics['num_queries_evaluated']}")
        
        print("\n✓ Comparison test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Augmented RAG Chunking System - Comprehensive Test Suite")
    print("=" * 80)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it to run paraphrase generation tests.")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    pdf_path = "/home/ubuntu/attachments/30af5be2-cb9a-4976-9c52-fcbb02d7b303/CRDS-FY2024-FR-09.pdf"
    
    results = []
    
    results.append(("Paraphrase Generation", test_paraphrase_generation()))
    
    results.append(("Augmented Chunking and RAG", test_augmented_chunking_and_rag(pdf_path, num_chunks_to_test=2)))
    
    results.append(("Comparison with Baseline", test_comparison_with_baseline(pdf_path, num_chunks_to_test=2)))
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠ {len(results) - total_passed} test(s) failed")

if __name__ == "__main__":
    main()
