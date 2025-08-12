"""
Utility functions for NLP Tool for YPAR
"""
import re
import hashlib
from typing import Any, Dict, List, Optional
import streamlit as st
from datetime import datetime
import uuid


def sanitize_input(text: str, max_length: int = 100000) -> str:
    """Sanitize user input text"""
    if not text:
        return ""
    
    # Truncate if too long
    text = text[:max_length]
    
    # Remove potentially harmful patterns
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    return text.strip()


def validate_file(file, max_size: int = 10 * 1024 * 1024) -> tuple[bool, str]:
    """Validate uploaded file"""
    if not file:
        return False, "No file provided"
    
    # Check file size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    
    if size > max_size:
        return False, f"File size ({size/1024/1024:.1f}MB) exceeds maximum ({max_size/1024/1024:.1f}MB)"
    
    # Check file type
    allowed_types = {
        'text/plain': 'txt',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx'
    }
    
    if file.type not in allowed_types:
        return False, f"Unsupported file type: {file.type}"
    
    return True, "File valid"


def generate_file_id(content: str, filename: str) -> str:
    """Generate unique file ID based on content and filename"""
    hash_input = f"{filename}_{content[:1000]}_{datetime.now().isoformat()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for processing"""
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to break at sentence boundary
        if end < text_len:
            last_period = text.rfind('.', start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end < text_len else end
    
    return chunks


def format_results(results: Any, result_type: str = "general") -> str:
    """Format analysis results for display"""
    if not results:
        return "No results available"
    
    if isinstance(results, dict):
        formatted = []
        for key, value in results.items():
            if isinstance(value, (list, dict)):
                formatted.append(f"**{key}**: {len(value)} items")
            else:
                formatted.append(f"**{key}**: {value}")
        return "\n\n".join(formatted)
    
    elif isinstance(results, list):
        return "\n\n".join([f"- {item}" for item in results])
    
    else:
        return str(results)


def cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key for function results"""
    key_parts = []
    
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        elif isinstance(arg, (list, tuple)):
            key_parts.append(str(hash(tuple(arg))))
        elif isinstance(arg, dict):
            key_parts.append(str(hash(tuple(sorted(arg.items())))))
        else:
            key_parts.append(str(id(arg)))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    return hashlib.md5("_".join(key_parts).encode()).hexdigest()


def safe_json_parse(json_str: str) -> Optional[Dict]:
    """Safely parse JSON string"""
    import json
    
    if not json_str:
        return None
    
    try:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', json_str)
        if json_match:
            json_str = json_match.group(1)
        
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)
        
        try:
            return json.loads(json_str)
        except:
            return None


def display_progress(message: str, total: int = 100):
    """Display progress bar helper"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update(current: int):
        progress = min(current / total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"{message}: {current}/{total}")
    
    return update


def get_word_statistics(text: str) -> Dict[str, Any]:
    """Get basic word statistics from text"""
    if not text:
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'unique_words': 0
        }
    
    # Basic tokenization
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'unique_words': len(set(words)),
        'vocabulary_richness': len(set(words)) / len(words) if words else 0
    }