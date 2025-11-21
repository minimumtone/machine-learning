import os
from typing import List, Optional
import json
from pathlib import Path
import hashlib
from openai import OpenAI
from chunking_strategies import Chunk

class ParaphraseGenerator:
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = ".paraphrase_cache"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, text: str, num_variations: int) -> str:
        content = f"{text}_{num_variations}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[str]]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, paraphrases: List[str]):
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(paraphrases, f, ensure_ascii=False, indent=2)
    
    def generate_paraphrases(self, text: str, num_variations: int = 10) -> List[str]:
        cache_key = self._get_cache_key(text, num_variations)
        cached_result = self._load_from_cache(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        prompt = f"""以下の文章を{num_variations}通りの異なる表現で言い換えてください。
元の意味を保ちながら、異なる言葉遣いや文構造を使用してください。

元の文章:
{text}

要件:
1. 元の意味を正確に保つこと
2. 各バリエーションは明確に異なる表現を使用すること
3. 自然な日本語であること
4. 専門用語は適切に保持すること
5. 各バリエーションは1行で出力すること

出力形式:
1. [バリエーション1]
2. [バリエーション2]
...
{num_variations}. [バリエーション{num_variations}]
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "あなたは日本語の言い換え専門家です。与えられた文章を様々な表現で言い換えることができます。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            
            paraphrases = []
            for line in content.split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, num_variations + 1)):
                    paraphrase = line.split('.', 1)[1].strip()
                    if paraphrase:
                        paraphrases.append(paraphrase)
            
            if len(paraphrases) < num_variations:
                print(f"Warning: Generated only {len(paraphrases)} paraphrases instead of {num_variations}")
            
            self._save_to_cache(cache_key, paraphrases)
            
            return paraphrases
            
        except Exception as e:
            print(f"Error generating paraphrases: {e}")
            return []
    
    def generate_paraphrases_batch(self, texts: List[str], num_variations: int = 10) -> List[List[str]]:
        results = []
        for i, text in enumerate(texts):
            print(f"Generating paraphrases for text {i+1}/{len(texts)}...")
            paraphrases = self.generate_paraphrases(text, num_variations)
            results.append(paraphrases)
        return results

class AugmentedChunk:
    def __init__(self, original_chunk: Chunk, paraphrases: List[str]):
        self.original_chunk = original_chunk
        self.paraphrases = paraphrases
        self.all_variations = [original_chunk.content] + paraphrases
    
    def __repr__(self):
        return f"AugmentedChunk(original_id={self.original_chunk.chunk_id}, variations={len(self.all_variations)})"

def augment_chunks(chunks: List[Chunk], num_variations: int = 10, api_key: Optional[str] = None) -> List[AugmentedChunk]:
    generator = ParaphraseGenerator(api_key=api_key)
    
    augmented_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Augmenting chunk {i+1}/{len(chunks)} (ID: {chunk.chunk_id})...")
        paraphrases = generator.generate_paraphrases(chunk.content, num_variations)
        augmented_chunk = AugmentedChunk(chunk, paraphrases)
        augmented_chunks.append(augmented_chunk)
    
    return augmented_chunks

if __name__ == "__main__":
    sample_text = "科学技術政策の動向について、主要国・地域の取り組みを分析し、日本の政策立案に活用することが重要です。"
    
    generator = ParaphraseGenerator()
    paraphrases = generator.generate_paraphrases(sample_text, num_variations=5)
    
    print("Original:")
    print(sample_text)
    print("\nParaphrases:")
    for i, p in enumerate(paraphrases, 1):
        print(f"{i}. {p}")
