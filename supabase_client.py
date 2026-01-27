"""
Supabase Client for PDF Compare System
Handles document storage and retrieval for database-based comparison
"""

import os
import json
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


class SupabaseClient:
    """Client for interacting with Supabase database"""
    
    def __init__(self, url=None, key=None):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and Key are required. Set SUPABASE_URL and SUPABASE_KEY in .env")
        
        self.client: Client = create_client(self.url, self.key)
        print("✅ Supabase client initialized")
    
    def upload_document(self, document_id: str, filename: str, extracted_text: str, 
                        word_index: dict, page_count: int) -> dict:
        """
        Upload a reference document to the database
        
        Args:
            document_id: Unique identifier (e.g., "P6100283.1", "PS006172")
            filename: Original filename
            extracted_text: Full OCR text for LLM comparison
            word_index: Word positions for highlighting (will be JSON serialized)
            page_count: Number of pages in document
        
        Returns:
            The inserted record
        """
        # Convert word_index to JSON-serializable format
        serialized_index = self._serialize_word_index(word_index)
        
        data = {
            "document_id": document_id,
            "filename": filename,
            "extracted_text": extracted_text,
            "word_index": serialized_index,
            "page_count": page_count
        }
        
        # Upsert to handle updates to existing documents
        result = self.client.table("documents").upsert(
            data, 
            on_conflict="document_id"
        ).execute()
        
        print(f"✅ Document '{document_id}' uploaded successfully")
        return result.data[0] if result.data else None
    
    def get_document_by_id(self, document_id: str) -> dict:
        """
        Fetch a document by its ID
        
        Args:
            document_id: Document identifier (e.g., "P6100283.1")
        
        Returns:
            Document record or None
        """
        result = self.client.table("documents").select("*").eq(
            "document_id", document_id
        ).execute()
        
        if result.data:
            doc = result.data[0]
            # Deserialize word_index back to original format
            doc['word_index'] = self._deserialize_word_index(doc.get('word_index', {}))
            return doc
        return None
    
    def list_documents(self) -> list:
        """
        List all stored reference documents
        
        Returns:
            List of documents with id, filename, and page_count
        """
        result = self.client.table("documents").select(
            "id, document_id, filename, page_count, created_at"
        ).order("created_at", desc=True).execute()
        
        return result.data or []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the database
        
        Args:
            document_id: Document identifier
        
        Returns:
            True if deleted, False otherwise
        """
        result = self.client.table("documents").delete().eq(
            "document_id", document_id
        ).execute()
        
        if result.data:
            print(f"✅ Document '{document_id}' deleted")
            return True
        return False
    
    def get_word_index(self, document_id: str) -> dict:
        """
        Fetch only the word index for a document (for highlighting)
        
        Args:
            document_id: Document identifier
        
        Returns:
            Word index dictionary or empty dict
        """
        result = self.client.table("documents").select(
            "word_index"
        ).eq("document_id", document_id).execute()
        
        if result.data:
            return self._deserialize_word_index(result.data[0].get('word_index', {}))
        return {}
    
    def get_extracted_text(self, document_id: str) -> str:
        """
        Fetch only the extracted text for a document (for LLM comparison)
        
        Args:
            document_id: Document identifier
        
        Returns:
            Extracted text or empty string
        """
        result = self.client.table("documents").select(
            "extracted_text"
        ).eq("document_id", document_id).execute()
        
        if result.data:
            return result.data[0].get('extracted_text', '')
        return ''
    
    def _serialize_word_index(self, word_index: dict) -> dict:
        """
        Serialize word_index for JSON storage in Supabase
        Converts tuple keys and values to lists
        """
        serialized = {
            'by_page': {},
            'by_word': {}
        }
        
        # Serialize by_page: {page_num: [(x0, y0, x1, y1, text, ...)]}
        for page_num, words in word_index.get('by_page', {}).items():
            serialized['by_page'][str(page_num)] = [
                list(word) if isinstance(word, tuple) else word 
                for word in words
            ]
        
        # Serialize by_word: {word: [(page_num, word_tuple)]}
        for word, locations in word_index.get('by_word', {}).items():
            serialized['by_word'][word] = [
                [loc[0], list(loc[1]) if isinstance(loc[1], tuple) else loc[1]]
                for loc in locations
            ]
        
        return serialized
    
    def _deserialize_word_index(self, serialized: dict) -> dict:
        """
        Deserialize word_index from JSON storage
        Converts lists back to tuples where needed
        """
        from collections import defaultdict
        
        deserialized = {
            'by_page': {},
            'by_word': defaultdict(list)
        }
        
        # Deserialize by_page
        for page_num_str, words in serialized.get('by_page', {}).items():
            page_num = int(page_num_str)
            deserialized['by_page'][page_num] = [
                tuple(word) if isinstance(word, list) else word 
                for word in words
            ]
        
        # Deserialize by_word
        for word, locations in serialized.get('by_word', {}).items():
            for loc in locations:
                page_num = loc[0]
                word_tuple = tuple(loc[1]) if isinstance(loc[1], list) else loc[1]
                deserialized['by_word'][word].append((page_num, word_tuple))
        
        return deserialized


# Test connection when run directly
if __name__ == "__main__":
    try:
        client = SupabaseClient()
        docs = client.list_documents()
        print(f"📄 Found {len(docs)} documents in database")
        for doc in docs:
            print(f"  - {doc['document_id']}: {doc['filename']} ({doc['page_count']} pages)")
    except Exception as e:
        print(f"❌ Error: {e}")
