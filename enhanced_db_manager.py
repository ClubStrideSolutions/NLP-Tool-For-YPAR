"""
Enhanced MongoDB Database Manager for NLP YPAR Tool
Efficient storage and retrieval of document analyses
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import hashlib
import streamlit as st

logger = logging.getLogger(__name__)


class EnhancedMongoManager:
    """Enhanced MongoDB manager with optimized document and analysis storage"""
    
    def __init__(self, connection_string: str = None):
        """Initialize with connection string"""
        self.client = None
        self.db = None
        self.connected = False
        self.cache = {}
        self.documents_cache = {}
        self.analysis_cache = {}
        
        # Get connection string
        self.connection_string = connection_string or self._get_connection_string()
        
        if self.connection_string:
            self._connect()
            self._setup_collections()
    
    def _get_connection_string(self) -> str:
        """Get connection string from various sources"""
        # Check session state first
        if 'mongodb_connection_string' in st.session_state:
            return st.session_state['mongodb_connection_string']
        
        # Check environment or config
        import os
        return os.getenv("CONNECTION_STRING", "")
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # Test connection
            self.db = self.client["nlp_ypar"]
            self.connected = True
            logger.info("MongoDB connected successfully")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            self.connected = False
    
    def _setup_collections(self):
        """Setup collections with proper indexes for efficiency"""
        if not self.connected or not self.db:
            return
        
        try:
            # Documents collection indexes
            docs_col = self.db["documents"]
            docs_col.create_index([("file_id", ASCENDING)], unique=True)
            docs_col.create_index([("filename", ASCENDING)])
            docs_col.create_index([("upload_timestamp", DESCENDING)])
            docs_col.create_index([("last_analyzed", DESCENDING)])
            
            # Analysis results collection indexes
            analysis_col = self.db["analysis_results"]
            analysis_col.create_index([("document_id", ASCENDING), ("analysis_type", ASCENDING)])
            analysis_col.create_index([("timestamp", DESCENDING)])
            analysis_col.create_index([("document_name", ASCENDING)])
            analysis_col.create_index([
                ("document_name", "text"),
                ("analysis_type", "text"),
                ("summary", "text")
            ])  # Text index for search
            
            # Analysis summary collection
            summary_col = self.db["analysis_summary"]
            summary_col.create_index([("document_id", ASCENDING)], unique=True)
            
            logger.info("MongoDB collections and indexes created")
        except Exception as e:
            logger.error(f"Error setting up collections: {e}")
    
    def store_document(self, file_id: str, filename: str, content: str, 
                      metadata: Dict[str, Any]) -> bool:
        """Store document with metadata"""
        try:
            # Calculate content hash for duplicate detection
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            doc_record = {
                "file_id": file_id,
                "filename": filename,
                "content_preview": content[:2000],  # Store first 2000 chars
                "content_hash": content_hash,
                "content_length": len(content),
                "upload_timestamp": datetime.now(),
                "metadata": {
                    "word_count": metadata.get("word_count", 0),
                    "char_count": metadata.get("char_count", 0),
                    "type": metadata.get("type", "unknown"),
                    "size": metadata.get("size", 0),
                    "pages": metadata.get("pages", None)
                },
                "analyses_performed": [],
                "last_analyzed": None,
                "tags": [],
                "user": st.session_state.get("user", "anonymous")
            }
            
            # Cache locally
            self.documents_cache[file_id] = doc_record
            
            if self.connected and self.db:
                # Check for duplicates
                existing = self.db["documents"].find_one({"content_hash": content_hash})
                if existing and existing["file_id"] != file_id:
                    logger.warning(f"Duplicate content detected: {filename}")
                
                # Upsert document
                self.db["documents"].update_one(
                    {"file_id": file_id},
                    {"$set": doc_record},
                    upsert=True
                )
                
                # Initialize summary record
                self.db["analysis_summary"].update_one(
                    {"document_id": file_id},
                    {
                        "$set": {
                            "document_name": filename,
                            "total_analyses": 0,
                            "analysis_types": {},
                            "last_updated": datetime.now()
                        }
                    },
                    upsert=True
                )
                
                return True
        except Exception as e:
            logger.error(f"Error storing document: {e}")
        
        return False
    
    def store_analysis(self, file_id: str, analysis_type: str, 
                      results: Dict[str, Any], filename: str = None,
                      processing_time: float = 0) -> Optional[str]:
        """Store analysis results with efficient structure"""
        try:
            # Get document info
            doc_info = self.documents_cache.get(file_id)
            if not doc_info and self.connected:
                doc_info = self.db["documents"].find_one({"file_id": file_id})
            
            # Prepare analysis summary
            summary = self._extract_summary(analysis_type, results)
            
            analysis_record = {
                "document_id": file_id,
                "document_name": filename or (doc_info["filename"] if doc_info else f"doc_{file_id[:8]}"),
                "analysis_type": analysis_type,
                "timestamp": datetime.now(),
                "processing_time": processing_time,
                "summary": summary,
                "results": results,
                "metadata": {
                    "word_count": doc_info["metadata"]["word_count"] if doc_info else 0,
                    "method": results.get("method", "default"),
                    "parameters": results.get("parameters", {})
                },
                "status": "completed",
                "user": st.session_state.get("user", "anonymous")
            }
            
            # Cache locally
            cache_key = f"{file_id}_{analysis_type}"
            self.analysis_cache[cache_key] = analysis_record
            
            if self.connected and self.db:
                # Store analysis
                result = self.db["analysis_results"].insert_one(analysis_record)
                
                # Update document record
                self.db["documents"].update_one(
                    {"file_id": file_id},
                    {
                        "$addToSet": {"analyses_performed": analysis_type},
                        "$set": {"last_analyzed": datetime.now()},
                        "$inc": {"analysis_count": 1}
                    }
                )
                
                # Update analysis summary
                self.db["analysis_summary"].update_one(
                    {"document_id": file_id},
                    {
                        "$inc": {
                            "total_analyses": 1,
                            f"analysis_types.{analysis_type}": 1
                        },
                        "$set": {
                            "last_updated": datetime.now(),
                            f"latest_{analysis_type}": summary
                        }
                    }
                )
                
                return str(result.inserted_id)
            
            # Fallback to session state
            if 'analysis_results' not in st.session_state:
                st.session_state.analysis_results = []
            st.session_state.analysis_results.append(analysis_record)
            
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")
        
        return None
    
    def _extract_summary(self, analysis_type: str, results: Dict) -> str:
        """Extract summary from analysis results"""
        summary = ""
        
        if analysis_type == "sentiment":
            summary = f"Sentiment: {results.get('sentiment', 'N/A')}, Score: {results.get('score', 0):.2f}"
        elif analysis_type == "themes":
            themes = results.get('themes', [])[:3]
            summary = f"Top themes: {', '.join(themes)}"
        elif analysis_type == "keywords":
            keywords = results.get('keywords', [])[:5]
            summary = f"Keywords: {', '.join(keywords)}"
        elif analysis_type == "entities":
            entities = results.get('entities', {})
            summary = f"Found {sum(len(v) for v in entities.values())} entities"
        else:
            summary = str(results)[:200] if results else "Analysis completed"
        
        return summary
    
    def get_document_analyses(self, file_id: str) -> List[Dict]:
        """Get all analyses for a document"""
        if self.connected and self.db:
            try:
                analyses = list(self.db["analysis_results"].find(
                    {"document_id": file_id},
                    {"_id": 0}
                ).sort("timestamp", -1))
                
                # Group by type for better display
                grouped = {}
                for analysis in analyses:
                    a_type = analysis["analysis_type"]
                    if a_type not in grouped:
                        grouped[a_type] = []
                    grouped[a_type].append(analysis)
                
                return grouped
            except Exception as e:
                logger.error(f"Error getting analyses: {e}")
        
        # Fallback to cache
        return {k: v for k, v in self.analysis_cache.items() if k.startswith(file_id)}
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents with analysis summary"""
        if self.connected and self.db:
            try:
                pipeline = [
                    {
                        "$lookup": {
                            "from": "analysis_summary",
                            "localField": "file_id",
                            "foreignField": "document_id",
                            "as": "summary"
                        }
                    },
                    {
                        "$unwind": {
                            "path": "$summary",
                            "preserveNullAndEmptyArrays": True
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "file_id": 1,
                            "filename": 1,
                            "upload_timestamp": 1,
                            "metadata": 1,
                            "analyses_performed": 1,
                            "last_analyzed": 1,
                            "total_analyses": "$summary.total_analyses",
                            "analysis_types": "$summary.analysis_types"
                        }
                    },
                    {"$sort": {"upload_timestamp": -1}}
                ]
                
                return list(self.db["documents"].aggregate(pipeline))
            except Exception as e:
                logger.error(f"Error getting documents: {e}")
        
        return list(self.documents_cache.values())
    
    def search_analyses(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search analyses with filters"""
        if self.connected and self.db:
            try:
                search_query = {}
                
                # Text search
                if query:
                    search_query["$text"] = {"$search": query}
                
                # Apply filters
                if filters:
                    if filters.get("document_id"):
                        search_query["document_id"] = filters["document_id"]
                    if filters.get("analysis_type"):
                        search_query["analysis_type"] = filters["analysis_type"]
                    if filters.get("date_from"):
                        search_query["timestamp"] = {"$gte": filters["date_from"]}
                    if filters.get("date_to"):
                        search_query.setdefault("timestamp", {})["$lte"] = filters["date_to"]
                
                results = list(self.db["analysis_results"].find(
                    search_query,
                    {"_id": 0, "results": 0}  # Exclude full results for performance
                ).sort("timestamp", -1).limit(100))
                
                return results
            except Exception as e:
                logger.error(f"Error searching: {e}")
        
        return []
    
    def get_analysis_stats(self) -> Dict:
        """Get overall analysis statistics"""
        stats = {
            "total_documents": 0,
            "total_analyses": 0,
            "analysis_types": {},
            "recent_activity": []
        }
        
        if self.connected and self.db:
            try:
                # Document count
                stats["total_documents"] = self.db["documents"].count_documents({})
                
                # Analysis count and types
                pipeline = [
                    {
                        "$group": {
                            "_id": "$analysis_type",
                            "count": {"$sum": 1}
                        }
                    }
                ]
                
                for item in self.db["analysis_results"].aggregate(pipeline):
                    stats["analysis_types"][item["_id"]] = item["count"]
                    stats["total_analyses"] += item["count"]
                
                # Recent activity
                stats["recent_activity"] = list(self.db["analysis_results"].find(
                    {},
                    {"_id": 0, "document_name": 1, "analysis_type": 1, "timestamp": 1, "summary": 1}
                ).sort("timestamp", -1).limit(10))
                
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
        
        return stats
    
    def export_analysis_history(self, file_id: str = None) -> Dict:
        """Export analysis history for reporting"""
        if self.connected and self.db:
            try:
                query = {"document_id": file_id} if file_id else {}
                
                history = list(self.db["analysis_results"].find(
                    query,
                    {"_id": 0}
                ).sort("timestamp", -1))
                
                return {
                    "export_date": datetime.now().isoformat(),
                    "document_id": file_id,
                    "total_analyses": len(history),
                    "analyses": history
                }
            except Exception as e:
                logger.error(f"Error exporting history: {e}")
        
        return {}
    
    def cleanup_old_analyses(self, days: int = 30) -> int:
        """Clean up old analysis records"""
        if self.connected and self.db:
            try:
                cutoff_date = datetime.now() - timedelta(days=days)
                result = self.db["analysis_results"].delete_many(
                    {"timestamp": {"$lt": cutoff_date}}
                )
                return result.deleted_count
            except Exception as e:
                logger.error(f"Error cleaning up: {e}")
        
        return 0