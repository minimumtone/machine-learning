from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import MeCab
import re
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: int
    method: str

class MarkdownChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk(self, markdown_text: str) -> List[Chunk]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on, 
            strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(markdown_text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = []
        chunk_id = 0
        for doc in md_header_splits:
            splits = text_splitter.split_text(doc.page_content)
            for split in splits:
                chunks.append(Chunk(
                    content=split,
                    metadata=doc.metadata,
                    chunk_id=chunk_id,
                    method="markdown"
                ))
                chunk_id += 1
        
        return chunks

class MeCabSemanticChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mecab = MeCab.Tagger()
        
    def _get_sentence_boundaries(self, text: str) -> List[int]:
        boundaries = [0]
        parsed = self.mecab.parse(text)
        
        lines = parsed.split('\n')
        pos = 0
        for line in lines:
            if line == 'EOS' or line == '':
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            surface = parts[0]
            features = parts[1].split(',')
            
            pos += len(surface)
            
            if features[0] in ['記号'] and surface in ['。', '！', '？', '\n']:
                if pos < len(text):
                    boundaries.append(pos)
        
        boundaries.append(len(text))
        return boundaries
    
    def _extract_semantic_units(self, text: str) -> List[str]:
        boundaries = self._get_sentence_boundaries(text)
        sentences = []
        for i in range(len(boundaries) - 1):
            sentence = text[boundaries[i]:boundaries[i+1]].strip()
            if sentence:
                sentences.append(sentence)
        return sentences
    
    def chunk(self, markdown_text: str) -> List[Chunk]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on, 
            strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(markdown_text)
        
        chunks = []
        chunk_id = 0
        
        for doc in md_header_splits:
            semantic_units = self._extract_semantic_units(doc.page_content)
            
            current_chunk = ""
            for unit in semantic_units:
                if len(current_chunk) + len(unit) <= self.chunk_size:
                    current_chunk += unit + " "
                else:
                    if current_chunk.strip():
                        chunks.append(Chunk(
                            content=current_chunk.strip(),
                            metadata=doc.metadata,
                            chunk_id=chunk_id,
                            method="mecab_semantic"
                        ))
                        chunk_id += 1
                    
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + unit + " "
            
            if current_chunk.strip():
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    metadata=doc.metadata,
                    chunk_id=chunk_id,
                    method="mecab_semantic"
                ))
                chunk_id += 1
        
        return chunks

class SelfRouteChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, similarity_threshold: float = 0.7):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.mecab = MeCab.Tagger()
        
    def _get_sentences(self, text: str) -> List[str]:
        parsed = self.mecab.parse(text)
        
        sentences = []
        current_sentence = ""
        
        lines = parsed.split('\n')
        for line in lines:
            if line == 'EOS' or line == '':
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            surface = parts[0]
            features = parts[1].split(',')
            
            current_sentence += surface
            
            if features[0] in ['記号'] and surface in ['。', '！', '？']:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def chunk(self, markdown_text: str) -> List[Chunk]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on, 
            strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(markdown_text)
        
        chunks = []
        chunk_id = 0
        
        for doc in md_header_splits:
            sentences = self._get_sentences(doc.page_content)
            
            if not sentences:
                continue
            
            current_chunk = sentences[0]
            
            for i in range(1, len(sentences)):
                sentence = sentences[i]
                
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    similarity = self._calculate_semantic_similarity(current_chunk, sentence)
                    
                    if similarity >= self.similarity_threshold:
                        current_chunk += " " + sentence
                    else:
                        chunks.append(Chunk(
                            content=current_chunk.strip(),
                            metadata={**doc.metadata, 'semantic_boundary': True},
                            chunk_id=chunk_id,
                            method="self_route"
                        ))
                        chunk_id += 1
                        
                        overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                        current_chunk = overlap_text + " " + sentence
                else:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        metadata={**doc.metadata, 'semantic_boundary': False},
                        chunk_id=chunk_id,
                        method="self_route"
                    ))
                    chunk_id += 1
                    
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
            
            if current_chunk.strip():
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    metadata=doc.metadata,
                    chunk_id=chunk_id,
                    method="self_route"
                ))
                chunk_id += 1
        
        return chunks
