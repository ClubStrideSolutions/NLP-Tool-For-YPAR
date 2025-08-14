"""
MongoDB Database Backup and Restore Utility
Supports pickle, JSON, and CSV export formats
"""

import os
import sys
import pickle
import json
import csv
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from pymongo import MongoClient
from bson import ObjectId, Binary
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MongoDBBackup:
    """Handle MongoDB backup and restore operations"""
    
    def __init__(self, connection_string: str = None):
        """Initialize with MongoDB connection string"""
        self.connection_string = "mongodb+srv://javbarrios89:mediasense@clustersense.nh1tclt.mongodb.net/"
        self.client = None
        self.db = None
        
    def connect(self, database_name: str = "nlp_tool"):
        """Connect to MongoDB"""
        try:
            if not self.connection_string:
                raise ValueError("No connection string provided")
                
            self.client = MongoClient("mongodb+srv://javbarrios89:mediasense@clustersense.nh1tclt.mongodb.net/", serverSelectionTimeoutMS=5000)
            self.client.server_info()  # Test connection
            self.db = self.client[database_name]
            logger.info(f"Connected to MongoDB database: {database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def serialize_document(self, doc: Dict) -> Dict:
        """Convert MongoDB document to serializable format"""
        if isinstance(doc, dict):
            result = {}
            for key, value in doc.items():
                if isinstance(value, ObjectId):
                    result[key] = {"$oid": str(value)}
                elif isinstance(value, Binary):
                    result[key] = {"$binary": value.hex()}
                elif isinstance(value, datetime):
                    result[key] = {"$date": value.isoformat()}
                elif isinstance(value, dict):
                    result[key] = self.serialize_document(value)
                elif isinstance(value, list):
                    result[key] = [self.serialize_document(item) if isinstance(item, dict) else item for item in value]
                else:
                    result[key] = value
            return result
        return doc
    
    def deserialize_document(self, doc: Dict) -> Dict:
        """Convert serialized document back to MongoDB format"""
        if isinstance(doc, dict):
            result = {}
            for key, value in doc.items():
                if isinstance(value, dict):
                    if "$oid" in value:
                        result[key] = ObjectId(value["$oid"])
                    elif "$binary" in value:
                        result[key] = Binary(bytes.fromhex(value["$binary"]))
                    elif "$date" in value:
                        result[key] = datetime.fromisoformat(value["$date"])
                    else:
                        result[key] = self.deserialize_document(value)
                elif isinstance(value, list):
                    result[key] = [self.deserialize_document(item) if isinstance(item, dict) else item for item in value]
                else:
                    result[key] = value
            return result
        return doc
    
    def backup_to_pickle(self, output_dir: str = "backups"):
        """Backup all collections to pickle files"""
        try:
            # Create output directory
            backup_dir = Path(output_dir)
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            # Get all collections
            collections = self.db.list_collection_names()
            logger.info(f"Found {len(collections)} collections to backup")
            
            backup_metadata = {
                "timestamp": timestamp,
                "database": self.db.name,
                "collections": {}
            }
            
            for collection_name in collections:
                collection = self.db[collection_name]
                documents = list(collection.find())
                
                # Serialize documents
                serialized_docs = [self.serialize_document(doc) for doc in documents]
                
                # Save to pickle
                pickle_file = backup_path / f"{collection_name}.pkl"
                with open(pickle_file, 'wb') as f:
                    pickle.dump(serialized_docs, f)
                
                backup_metadata["collections"][collection_name] = {
                    "count": len(serialized_docs),
                    "file": f"{collection_name}.pkl"
                }
                
                logger.info(f"Backed up {len(documents)} documents from {collection_name}")
            
            # Save metadata
            metadata_file = backup_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            logger.info(f"Backup completed successfully at {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise
    
    def backup_to_json(self, output_dir: str = "backups"):
        """Backup all collections to JSON files"""
        try:
            backup_dir = Path(output_dir)
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"backup_json_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            collections = self.db.list_collection_names()
            
            for collection_name in collections:
                collection = self.db[collection_name]
                documents = list(collection.find())
                
                # Serialize documents
                serialized_docs = [self.serialize_document(doc) for doc in documents]
                
                # Save to JSON
                json_file = backup_path / f"{collection_name}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(serialized_docs, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Exported {len(documents)} documents from {collection_name} to JSON")
            
            logger.info(f"JSON backup completed at {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"JSON backup failed: {e}")
            raise
    
    def backup_to_csv(self, output_dir: str = "backups"):
        """Backup collections to CSV files (for tabular data)"""
        try:
            backup_dir = Path(output_dir)
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"backup_csv_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            collections = self.db.list_collection_names()
            
            for collection_name in collections:
                collection = self.db[collection_name]
                documents = list(collection.find())
                
                if documents:
                    # Convert to DataFrame
                    df = pd.DataFrame(documents)
                    
                    # Convert ObjectId to string
                    if '_id' in df.columns:
                        df['_id'] = df['_id'].astype(str)
                    
                    # Save to CSV
                    csv_file = backup_path / f"{collection_name}.csv"
                    df.to_csv(csv_file, index=False, encoding='utf-8')
                    
                    logger.info(f"Exported {len(documents)} documents from {collection_name} to CSV")
            
            logger.info(f"CSV backup completed at {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"CSV backup failed: {e}")
            raise
    
    def restore_from_pickle(self, backup_path: str, clear_existing: bool = False):
        """Restore database from pickle backup"""
        try:
            backup_dir = Path(backup_path)
            
            if not backup_dir.exists():
                raise ValueError(f"Backup directory not found: {backup_path}")
            
            # Load metadata
            metadata_file = backup_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Restoring backup from {metadata['timestamp']}")
            
            # Get pickle files
            pickle_files = list(backup_dir.glob("*.pkl"))
            
            for pickle_file in pickle_files:
                collection_name = pickle_file.stem
                collection = self.db[collection_name]
                
                # Clear existing data if requested
                if clear_existing:
                    collection.delete_many({})
                    logger.info(f"Cleared existing data in {collection_name}")
                
                # Load and restore documents
                with open(pickle_file, 'rb') as f:
                    documents = pickle.load(f)
                
                # Deserialize documents
                deserialized_docs = [self.deserialize_document(doc) for doc in documents]
                
                if deserialized_docs:
                    collection.insert_many(deserialized_docs)
                    logger.info(f"Restored {len(deserialized_docs)} documents to {collection_name}")
            
            logger.info("Restore completed successfully")
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            raise
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="MongoDB Backup and Restore Utility")
    
    parser.add_argument(
        "action",
        choices=["backup", "restore"],
        help="Action to perform"
    )
    
    parser.add_argument(
        "--connection-string",
        help="MongoDB connection string (or set CONNECTION_STRING env var)"
    )
    
    parser.add_argument(
        "--database",
        default="nlp_tool",
        help="Database name (default: nlp_tool)"
    )
    
    parser.add_argument(
        "--format",
        choices=["pickle", "json", "csv", "all"],
        default="pickle",
        help="Backup format (default: pickle)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="backups",
        help="Output directory for backups (default: backups)"
    )
    
    parser.add_argument(
        "--backup-path",
        help="Path to backup directory for restore"
    )
    
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing data before restore"
    )
    
    args = parser.parse_args()
    
    # Initialize backup manager
    backup_manager = MongoDBBackup(args.connection_string)
    
    # Connect to database
    if not backup_manager.connect(args.database):
        logger.error("Failed to connect to database")
        sys.exit(1)
    
    try:
        if args.action == "backup":
            # Perform backup
            if args.format == "pickle":
                path = backup_manager.backup_to_pickle(args.output_dir)
                print(f"✅ Pickle backup saved to: {path}")
            elif args.format == "json":
                path = backup_manager.backup_to_json(args.output_dir)
                print(f"✅ JSON backup saved to: {path}")
            elif args.format == "csv":
                path = backup_manager.backup_to_csv(args.output_dir)
                print(f"✅ CSV backup saved to: {path}")
            elif args.format == "all":
                pickle_path = backup_manager.backup_to_pickle(args.output_dir)
                json_path = backup_manager.backup_to_json(args.output_dir)
                csv_path = backup_manager.backup_to_csv(args.output_dir)
                print(f"✅ Backups saved to:")
                print(f"   Pickle: {pickle_path}")
                print(f"   JSON: {json_path}")
                print(f"   CSV: {csv_path}")
        
        elif args.action == "restore":
            # Perform restore
            if not args.backup_path:
                logger.error("--backup-path required for restore")
                sys.exit(1)
            
            backup_manager.restore_from_pickle(args.backup_path, args.clear_existing)
            print(f"✅ Database restored from: {args.backup_path}")
    
    finally:
        backup_manager.close()


if __name__ == "__main__":
    main()