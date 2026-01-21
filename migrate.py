"""
MongoDB Data Migration Script

Automatically migrates data from external MongoDB Atlas to internal Docker MongoDB.
Runs on startup and only migrates if the internal database is empty.
"""

import os
import sys
import logging
from pymongo import MongoClient
from dotenv import load_dotenv
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# External MongoDB Atlas URL (your original connection)
EXTERNAL_MONGODB_URL = "mongodb+srv://puspenduofficial:4xwjcNbgRv6QVA0H@predictdata.hfw7l.mongodb.net/"

# Internal MongoDB URL (Docker)
INTERNAL_MONGODB_URL = os.getenv(
    "MONGODB_URL", 
    "mongodb://predictuser:predictpass123@mongodb:27017/prediction_db?authSource=admin"
)

DATABASE_NAME = os.getenv("DATABASE_NAME", "prediction_db")
COLLECTIONS = ["results", "prediction_history", "model_accuracy"]


def wait_for_mongodb(url: str, max_retries: int = 30, delay: int = 2) -> MongoClient:
    """Wait for MongoDB to be ready and return client."""
    for attempt in range(max_retries):
        try:
            client = MongoClient(url, serverSelectionTimeoutMS=5000)
            client.server_info()  # Test connection
            logger.info(f"Connected to MongoDB: {url[:50]}...")
            return client
        except Exception as e:
            logger.warning(f"MongoDB not ready (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(delay)
    
    raise Exception(f"Could not connect to MongoDB after {max_retries} attempts")


def check_internal_db_empty(internal_client: MongoClient) -> bool:
    """Check if internal database is empty (needs migration)."""
    db = internal_client[DATABASE_NAME]
    
    total_docs = 0
    for collection_name in COLLECTIONS:
        if collection_name in db.list_collection_names():
            count = db[collection_name].count_documents({})
            total_docs += count
            logger.info(f"Internal collection '{collection_name}' has {count} documents")
    
    return total_docs == 0


def migrate_collection(
    external_db, 
    internal_db, 
    collection_name: str,
    batch_size: int = 1000
) -> int:
    """Migrate a single collection from external to internal DB."""
    external_col = external_db[collection_name]
    internal_col = internal_db[collection_name]
    
    total_docs = external_col.count_documents({})
    if total_docs == 0:
        logger.info(f"Collection '{collection_name}' is empty, skipping")
        return 0
    
    logger.info(f"Migrating {total_docs} documents from '{collection_name}'...")
    
    migrated = 0
    cursor = external_col.find({})
    batch = []
    
    for doc in cursor:
        # Remove _id to let MongoDB generate new ones
        doc.pop('_id', None)
        batch.append(doc)
        
        if len(batch) >= batch_size:
            try:
                internal_col.insert_many(batch, ordered=False)
                migrated += len(batch)
                logger.info(f"  Migrated {migrated}/{total_docs} documents...")
            except Exception as e:
                logger.warning(f"  Batch insert warning: {e}")
            batch = []
    
    # Insert remaining documents
    if batch:
        try:
            internal_col.insert_many(batch, ordered=False)
            migrated += len(batch)
        except Exception as e:
            logger.warning(f"  Final batch warning: {e}")
    
    logger.info(f"Completed migration of '{collection_name}': {migrated} documents")
    return migrated


def run_migration():
    """Main migration function."""
    logger.info("=" * 60)
    logger.info("Starting MongoDB Data Migration")
    logger.info("=" * 60)
    
    try:
        # Connect to internal MongoDB first (wait for Docker container)
        logger.info("Connecting to internal MongoDB (Docker)...")
        internal_client = wait_for_mongodb(INTERNAL_MONGODB_URL)
        internal_db = internal_client[DATABASE_NAME]
        
        # Check if migration is needed
        if not check_internal_db_empty(internal_client):
            logger.info("Internal database already has data. Skipping migration.")
            logger.info("To force migration, clear the internal database first.")
            internal_client.close()
            return True
        
        # Connect to external MongoDB
        logger.info("Connecting to external MongoDB (Atlas)...")
        try:
            external_client = wait_for_mongodb(EXTERNAL_MONGODB_URL, max_retries=5)
            external_db = external_client[DATABASE_NAME]
        except Exception as e:
            logger.warning(f"Could not connect to external MongoDB: {e}")
            logger.info("Starting with empty database (no migration).")
            internal_client.close()
            return True
        
        # Migrate each collection
        total_migrated = 0
        for collection_name in COLLECTIONS:
            try:
                count = migrate_collection(external_db, internal_db, collection_name)
                total_migrated += count
            except Exception as e:
                logger.error(f"Error migrating collection '{collection_name}': {e}")
        
        logger.info("=" * 60)
        logger.info(f"Migration complete! Total documents migrated: {total_migrated}")
        logger.info("=" * 60)
        
        # Close connections
        external_client.close()
        internal_client.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
