"""
OpenAI-powered Analysis Module for NLP YPAR Tool
Uses GPT models for intelligent text analysis
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class OpenAIAnalyzer:
    """OpenAI-powered text analysis"""
    
    def __init__(self):
        """Initialize with API key from session state"""
        self.api_key = st.session_state.get('openai_api_key', '')
        self.client = None
        self.model = "gpt-3.5-turbo"  # Default model, can upgrade to gpt-4
        self.initialized = False
        
        if self.api_key:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.initialized = True
            logger.info("OpenAI client initialized successfully")
        except ImportError:
            logger.error("OpenAI library not installed. Run: pip install openai")
            st.error("Please install OpenAI library: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            st.error(f"OpenAI initialization failed: {str(e)[:100]}")
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return self.initialized and self.client is not None
    
    def analyze_sentiment(self, text: str, detailed: bool = True) -> Dict[str, Any]:
        """AI-powered sentiment analysis with context understanding"""
        if not self.is_available():
            return {"error": "OpenAI not available"}
        
        try:
            # Truncate text if too long
            text_sample = text[:4000] if len(text) > 4000 else text
            
            prompt = f"""Analyze the sentiment of the following text. Provide:
1. Overall sentiment (positive/negative/neutral/mixed)
2. Sentiment score (-1 to 1)
3. Key emotional themes identified
4. Confidence level (0-100%)
5. Brief explanation of the sentiment

Text: {text_sample}

Return response as JSON with keys: sentiment, score, themes, confidence, explanation"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sentiment analyst. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            
            # Parse JSON response
            try:
                parsed = json.loads(result)
                return {
                    "sentiment": parsed.get("sentiment", "neutral"),
                    "score": float(parsed.get("score", 0)),
                    "themes": parsed.get("themes", []),
                    "confidence": parsed.get("confidence", 0),
                    "explanation": parsed.get("explanation", ""),
                    "method": "OpenAI GPT",
                    "model": self.model
                }
            except json.JSONDecodeError:
                # Fallback parsing
                return {
                    "sentiment": "analyzed",
                    "score": 0,
                    "explanation": result,
                    "method": "OpenAI GPT",
                    "model": self.model
                }
                
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"error": str(e)}
    
    def extract_themes(self, text: str, num_themes: int = 5) -> Dict[str, Any]:
        """AI-powered theme extraction with explanations"""
        if not self.is_available():
            return {"error": "OpenAI not available"}
        
        try:
            text_sample = text[:6000] if len(text) > 6000 else text
            
            prompt = f"""Analyze the following text and extract the {num_themes} main themes or topics. For each theme provide:
1. Theme title (2-4 words)
2. Description (1-2 sentences)
3. Key related concepts/keywords (5-10 words)
4. Relevance score (0-100%)
5. Supporting quotes from the text (1-2 short quotes)

Text: {text_sample}

Return as JSON with key 'themes' containing an array of theme objects."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at thematic analysis and topic extraction. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            
            try:
                parsed = json.loads(result)
                return {
                    "themes": parsed.get("themes", []),
                    "method": "OpenAI GPT",
                    "model": self.model,
                    "num_themes": num_themes
                }
            except:
                return {
                    "themes": [],
                    "raw_response": result,
                    "method": "OpenAI GPT",
                    "model": self.model
                }
                
        except Exception as e:
            logger.error(f"Theme extraction error: {e}")
            return {"error": str(e)}
    
    def extract_keywords(self, text: str, num_keywords: int = 20) -> Dict[str, Any]:
        """AI-powered keyword extraction with importance ranking"""
        if not self.is_available():
            return {"error": "OpenAI not available"}
        
        try:
            text_sample = text[:4000] if len(text) > 4000 else text
            
            prompt = f"""Extract the {num_keywords} most important keywords and key phrases from this text. 
For each keyword provide:
1. The keyword/phrase
2. Importance score (0-100)
3. Category (person, place, concept, organization, etc.)
4. Context or definition (1 sentence)

Text: {text_sample}

Return as JSON with key 'keywords' containing an array of keyword objects."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at keyword extraction and text analysis. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            result = response.choices[0].message.content
            
            try:
                parsed = json.loads(result)
                return {
                    "keywords": parsed.get("keywords", []),
                    "method": "OpenAI GPT",
                    "model": self.model
                }
            except:
                return {
                    "keywords": [],
                    "raw_response": result,
                    "method": "OpenAI GPT"
                }
                
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return {"error": str(e)}
    
    def summarize_text(self, text: str, max_length: int = 500) -> Dict[str, Any]:
        """AI-powered text summarization"""
        if not self.is_available():
            return {"error": "OpenAI not available"}
        
        try:
            text_sample = text[:8000] if len(text) > 8000 else text
            
            prompt = f"""Provide a comprehensive summary of the following text in approximately {max_length} words. 
Include:
1. Main points and key arguments
2. Important findings or conclusions
3. Notable quotes or statements
4. Overall context and significance

Text: {text_sample}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating clear, concise summaries while preserving key information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=max_length * 2  # Tokens ≈ 1.5x words
            )
            
            summary = response.choices[0].message.content
            
            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": f"{(1 - len(summary)/len(text)) * 100:.1f}%",
                "method": "OpenAI GPT",
                "model": self.model
            }
                
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {"error": str(e)}
    
    def extract_quotes(self, text: str, num_quotes: int = 10) -> Dict[str, Any]:
        """AI-powered quote extraction with context"""
        if not self.is_available():
            return {"error": "OpenAI not available"}
        
        try:
            text_sample = text[:6000] if len(text) > 6000 else text
            
            prompt = f"""Extract {num_quotes} most significant quotes or statements from this text. 
For each quote provide:
1. The exact quote
2. Speaker/source (if identifiable)
3. Context (what it's about)
4. Significance (why it's important)
5. Sentiment (positive/negative/neutral)

Text: {text_sample}

Return as JSON with key 'quotes' containing an array of quote objects."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying significant quotes and statements. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            
            try:
                parsed = json.loads(result)
                return {
                    "quotes": parsed.get("quotes", []),
                    "method": "OpenAI GPT",
                    "model": self.model
                }
            except:
                return {
                    "quotes": [],
                    "raw_response": result,
                    "method": "OpenAI GPT"
                }
                
        except Exception as e:
            logger.error(f"Quote extraction error: {e}")
            return {"error": str(e)}
    
    def generate_insights(self, text: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """Generate research insights and recommendations"""
        if not self.is_available():
            return {"error": "OpenAI not available"}
        
        try:
            text_sample = text[:6000] if len(text) > 6000 else text
            
            focus_prompt = ""
            if focus_areas:
                focus_prompt = f"Focus particularly on these areas: {', '.join(focus_areas)}"
            
            prompt = f"""Analyze this text and provide research insights. {focus_prompt}

Provide:
1. Key findings (3-5 main discoveries)
2. Patterns identified
3. Recommendations for further research
4. Potential implications
5. Questions raised by the analysis
6. Strengths and limitations observed

Text: {text_sample}

Return as structured JSON with appropriate keys."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research analyst expert at generating actionable insights. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=1200
            )
            
            result = response.choices[0].message.content
            
            try:
                parsed = json.loads(result)
                return {
                    "insights": parsed,
                    "method": "OpenAI GPT",
                    "model": self.model,
                    "focus_areas": focus_areas
                }
            except:
                return {
                    "insights": result,
                    "method": "OpenAI GPT",
                    "model": self.model
                }
                
        except Exception as e:
            logger.error(f"Insight generation error: {e}")
            return {"error": str(e)}
    
    def answer_question(self, text: str, question: str) -> Dict[str, Any]:
        """Answer questions about the text using AI"""
        if not self.is_available():
            return {"error": "OpenAI not available"}
        
        try:
            text_sample = text[:6000] if len(text) > 6000 else text
            
            prompt = f"""Based on the following text, answer this question: {question}

Provide:
1. Direct answer
2. Supporting evidence from the text
3. Confidence level (0-100%)
4. Any caveats or limitations

Text: {text_sample}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing texts and answering questions accurately based on the provided content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            return {
                "question": question,
                "answer": answer,
                "method": "OpenAI GPT",
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Question answering error: {e}")
            return {"error": str(e)}
    
    def compare_documents(self, doc1: str, doc2: str) -> Dict[str, Any]:
        """Compare two documents using AI"""
        if not self.is_available():
            return {"error": "OpenAI not available"}
        
        try:
            doc1_sample = doc1[:3000] if len(doc1) > 3000 else doc1
            doc2_sample = doc2[:3000] if len(doc2) > 3000 else doc2
            
            prompt = f"""Compare these two documents and provide:
1. Main similarities
2. Key differences
3. Themes unique to each document
4. Overall comparison summary

Document 1: {doc1_sample}

Document 2: {doc2_sample}

Return as structured JSON."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at document comparison and analysis. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            
            result = response.choices[0].message.content
            
            try:
                parsed = json.loads(result)
                return {
                    "comparison": parsed,
                    "method": "OpenAI GPT",
                    "model": self.model
                }
            except:
                return {
                    "comparison": result,
                    "method": "OpenAI GPT",
                    "model": self.model
                }
                
        except Exception as e:
            logger.error(f"Document comparison error: {e}")
            return {"error": str(e)}
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics if available"""
        # This would need actual implementation based on OpenAI's usage API
        return {
            "api_key_configured": bool(self.api_key),
            "model": self.model,
            "initialized": self.initialized
        }