"""
Document Chunking Module for Mazag RAG System
Splits documents into semantically meaningful chunks for embedding and retrieval.
"""

from typing import List, Dict, Any
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_id: str
    metadata: Dict[str, Any]
    start_idx: int
    end_idx: int


class DocumentChunker:
    """
    Handles document chunking with multiple strategies.
    Supports fixed-size, sentence-based, semantic, and recursive chunking.
    """
    
    def __init__(
        self, 
        chunk_size: int = 512,
        overlap: int = 50,
        strategy: str = "fixed"
    ):
        """
        Args:
            chunk_size: Target size for each chunk (in characters)
            overlap: Number of overlapping WORDS between chunks (changed from characters)
            strategy: Chunking strategy ('fixed', 'sentence', 'semantic', 'recursive')
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
    
    def _words_to_chars(self, text: str, num_words: int) -> int:
        """
        Convert word count to approximate character count in the text.
        
        Args:
            text: The text to analyze
            num_words: Number of words to convert
            
        Returns:
            Approximate number of characters corresponding to num_words
        """
        if not text or num_words <= 0:
            return 0
        
        words = text.split()
        if len(words) <= num_words:
            return len(text)
        
        # Get the last num_words words and calculate their length
        overlap_words = words[-num_words:]
        overlap_text = " ".join(overlap_words)
        return len(overlap_text)
    
    def _get_overlap_text(self, text: str, num_words: int) -> str:
        """
        Get the last num_words from text for overlap.
        
        Args:
            text: The text to extract from
            num_words: Number of words to extract
            
        Returns:
            The last num_words as a string
        """
        if not text or num_words <= 0:
            return ""
        
        words = text.split()
        if len(words) <= num_words:
            return text
        
        return " ".join(words[-num_words:])
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """
        Split text into chunks based on the selected strategy.
        
        Args:
            text: The text to chunk
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        if metadata is None:
            metadata = {}
        
        if self.strategy == "fixed":
            return self._fixed_size_chunking(text, metadata)
        elif self.strategy == "sentence":
            return self._sentence_based_chunking(text, metadata)
        elif self.strategy == "semantic":
            return self._semantic_chunking(text, metadata)
        elif self.strategy == "recursive":
            return self._recursive_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def _fixed_size_chunking(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Fixed-size chunking with word-based overlap"""
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            # Don't create empty chunks
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"chunk_{chunk_idx}",
                    metadata={**metadata, "chunk_idx": chunk_idx},
                    start_idx=start,
                    end_idx=end
                ))
                chunk_idx += 1
            
            # Calculate overlap in characters based on word count
            overlap_chars = self._words_to_chars(chunk_text, self.overlap)
            start += self.chunk_size - overlap_chars
        
        return chunks
    
    def _sentence_based_chunking(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk by sentences, respecting chunk size limits with word-based overlap"""
        # Split into sentences (handles Arabic and English)
        sentence_endings = r'[.!?؟।।]+'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        start_idx = 0
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                end_idx = start_idx + len(current_chunk)
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    chunk_id=f"chunk_{chunk_idx}",
                    metadata={**metadata, "chunk_idx": chunk_idx},
                    start_idx=start_idx,
                    end_idx=end_idx
                ))
                chunk_idx += 1
                
                # Keep overlap from previous chunk (word-based)
                overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                start_idx = end_idx - len(overlap_text)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_idx}",
                metadata={**metadata, "chunk_idx": chunk_idx},
                start_idx=start_idx,
                end_idx=start_idx + len(current_chunk)
            ))
        
        return chunks
    
    def _semantic_chunking(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Semantic chunking based on paragraph breaks and topic changes.
        Falls back to sentence-based if no clear semantic boundaries.
        """
        # Split by paragraphs (double newlines or major breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        start_idx = 0
        
        for para in paragraphs:
            # If paragraph alone is too large, split it
            if len(para) > self.chunk_size:
                # Save current chunk if exists
                if current_chunk.strip():
                    end_idx = start_idx + len(current_chunk)
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        chunk_id=f"chunk_{chunk_idx}",
                        metadata={**metadata, "chunk_idx": chunk_idx},
                        start_idx=start_idx,
                        end_idx=end_idx
                    ))
                    chunk_idx += 1
                    current_chunk = ""
                
                # Split large paragraph using sentence-based
                para_chunks = self._sentence_based_chunking(
                    para, 
                    {**metadata, "from_large_para": True}
                )
                chunks.extend(para_chunks)
                start_idx = chunks[-1].end_idx
            
            # Add paragraph to current chunk if it fits
            elif len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            
            # Otherwise, save current and start new chunk
            else:
                if current_chunk.strip():
                    end_idx = start_idx + len(current_chunk)
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        chunk_id=f"chunk_{chunk_idx}",
                        metadata={**metadata, "chunk_idx": chunk_idx},
                        start_idx=start_idx,
                        end_idx=end_idx
                    ))
                    chunk_idx += 1
                    start_idx = end_idx
                
                current_chunk = para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                text=current_chunk.strip(),
                chunk_id=f"chunk_{chunk_idx}",
                metadata={**metadata, "chunk_idx": chunk_idx},
                start_idx=start_idx,
                end_idx=start_idx + len(current_chunk)
            ))
        
        return chunks
    
    def _recursive_chunking(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Recursive chunking with prioritized separators.
        Splits text using a hierarchy of separators, recursively splitting
        chunks that are still too large.
        
        Separator priority (highest to lowest):
        1. Double newlines (paragraphs)
        2. Single newlines (lines)
        3. Sentence endings (.!?؟)
        4. Commas and semicolons
        5. Spaces (words)
        
        Args:
            text: The text to chunk
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        # Define separators in priority order (highest priority first)
        separators = [
            "\n\n",      # Paragraphs
            "\n",        # Lines
            ". ",        # Sentences (with space to preserve period)
            "! ",        # Exclamations
            "? ",        # Questions
            "؟ ",        # Arabic question mark
            "; ",        # Semicolons
            ", ",        # Commas
            " ",         # Words
        ]
        
        # Start recursive splitting
        chunks = self._split_text_recursive(text, separators, metadata)
        
        # Assign chunk IDs and indices
        final_chunks = []
        for idx, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                final_chunks.append(Chunk(
                    text=chunk_text.strip(),
                    chunk_id=f"chunk_{idx}",
                    metadata={**metadata, "chunk_idx": idx, "strategy": "recursive"},
                    start_idx=0,  # Could be calculated from original text if needed
                    end_idx=len(chunk_text)
                ))
        
        return final_chunks
    
    def _split_text_recursive(
        self,
        text: str,
        separators: List[str],
        metadata: Dict[str, Any],
        depth: int = 0
    ) -> List[str]:
        """
        Recursively split text using the separator hierarchy.
        
        Args:
            text: Text to split
            separators: List of separators to try (in priority order)
            metadata: Chunk metadata
            depth: Current recursion depth (for debugging)
            
        Returns:
            List of text chunks
        """
        # Base case: text is small enough
        if len(text) <= self.chunk_size:
            return [text]
        
        # Base case: no more separators to try
        if not separators:
            # Force split by chunk_size with word-based overlap
            return self._force_split(text)
        
        # Try to split with the current separator
        current_separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        splits = text.split(current_separator)
        
        # If we couldn't split (only one piece), try next separator
        if len(splits) == 1:
            return self._split_text_recursive(text, remaining_separators, metadata, depth + 1)
        
        # Merge splits into chunks, respecting chunk_size
        chunks = []
        current_chunk = ""
        
        for i, split in enumerate(splits):
            # Reconstruct the separator (except for last split)
            if i < len(splits) - 1:
                split_with_sep = split + current_separator
            else:
                split_with_sep = split
            
            # If this split alone is too large, recursively split it
            if len(split_with_sep) > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Recursively split the large piece
                sub_chunks = self._split_text_recursive(
                    split_with_sep,
                    remaining_separators,
                    metadata,
                    depth + 1
                )
                
                # Add sub-chunks with word-based overlap
                for j, sub_chunk in enumerate(sub_chunks):
                    if j > 0 and chunks:
                        # Add overlap from previous chunk
                        overlap_text = self._get_overlap_text(chunks[-1], self.overlap)
                        if overlap_text:
                            sub_chunk = overlap_text + " " + sub_chunk
                    chunks.append(sub_chunk.strip())
                
            # If adding this split would exceed chunk_size, save current chunk
            elif current_chunk and len(current_chunk) + len(split_with_sep) > self.chunk_size:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous
                overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                current_chunk = overlap_text + " " + split_with_sep if overlap_text else split_with_sep
            
            # Otherwise, add to current chunk
            else:
                if current_chunk:
                    current_chunk += split_with_sep
                else:
                    current_chunk = split_with_sep
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _force_split(self, text: str) -> List[str]:
        """
        Force split text by chunk_size when no separator works.
        Uses word-based overlap.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Calculate overlap in characters based on word count
            overlap_chars = self._words_to_chars(chunk_text, self.overlap)
            start += self.chunk_size - overlap_chars
        
        return chunks
    
    def chunk_conversation(
        self, 
        messages: List[Dict[str, str]]
    ) -> List[Chunk]:
        """
        Chunk a conversation history.
        Groups messages into meaningful chunks while preserving context.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            List of conversation chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        
        for msg in messages:
            msg_text = f"{msg['role']}: {msg['content']}"
            msg_length = len(msg_text)
            
            # If adding this message exceeds chunk size, save current chunk
            if current_length + msg_length > self.chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"conv_chunk_{chunk_idx}",
                    metadata={
                        "type": "conversation",
                        "num_messages": len(current_chunk),
                        "chunk_idx": chunk_idx
                    },
                    start_idx=0,
                    end_idx=len(chunk_text)
                ))
                chunk_idx += 1
                
                # Keep last message for context
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_length = len(current_chunk[0]) if current_chunk else 0
            
            current_chunk.append(msg_text)
            current_length += msg_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"conv_chunk_{chunk_idx}",
                metadata={
                    "type": "conversation",
                    "num_messages": len(current_chunk),
                    "chunk_idx": chunk_idx
                },
                start_idx=0,
                end_idx=len(chunk_text)
            ))
        
        return chunks


# Utility functions
def chunk_document(
    text: str, 
    chunk_size: int = 512, 
    overlap: int = 5,
    strategy: str = "semantic"
) -> List[str]:
    """
    Quick utility to chunk a document and return text chunks only.
    
    Args:
        text: Document text
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        strategy: Chunking strategy
        
    Returns:
        List of text chunks
    """
    chunker = DocumentChunker(chunk_size, overlap, strategy)
    chunks = chunker.chunk_text(text)
    return [chunk.text for chunk in chunks]

