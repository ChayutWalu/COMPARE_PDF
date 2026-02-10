"""
Upload Reference Documents to Supabase
Script to upload PDF files to the database for comparison
"""

import os
import sys
import re
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from supabase_client import SupabaseClient
from ocr_processor import extract_text_from_pdf, build_word_index_parallel, build_word_index, PARALLEL_OCR, FORCE_OCR
import fitz


def extract_document_id(filepath: str) -> str:
    """
    Extract document ID from filename
    Examples:
        P6100283.1.pdf -> P6100283.1
        PS006172.pdf -> PS006172
        P5800133.pdf -> P5800133
    """
    filename = os.path.basename(filepath)
    # Remove .pdf extension
    name = os.path.splitext(filename)[0]
    return name


def upload_document(pdf_path: str, document_id: str = None, progress_callback=None) -> bool:
    """
    Upload a PDF document to Supabase
    
    Args:
        pdf_path: Path to PDF file
        document_id: Optional custom document ID (auto-extracted from filename if not provided)
        progress_callback: Optional callback for progress updates
    
    Returns:
        True if successful, False otherwise
    """
    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)
    
    if not os.path.exists(pdf_path):
        log(f"❌ File not found: {pdf_path}")
        return False
    
    # Extract document ID from filename if not provided
    if not document_id:
        document_id = extract_document_id(pdf_path)
    
    filename = os.path.basename(pdf_path)
    log(f"📄 Processing: {filename}")
    log(f"   Document ID: {document_id}")
    
    try:
        # Extract text for LLM comparison
        log("   → Extracting text...")
        extracted_text, ocr_engine = extract_text_from_pdf(pdf_path)
        log(f"   ✅ อ่านข้อความด้วย: PaddleOCR (Local)")
        
        # Build word index for highlighting
        log("   → Building word index (OCR)...")
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        
        if PARALLEL_OCR and FORCE_OCR and page_count > 1:
            doc.close()
            word_index = build_word_index_parallel(pdf_path, progress_callback=lambda msg, pct: log(f"   {msg}"))
        else:
            word_index = build_word_index(doc, progress_callback=lambda msg, pct: log(f"   {msg}"))
            doc.close()
        
        # Upload to Supabase
        log("   → Uploading to Supabase...")
        client = SupabaseClient()
        result = client.upload_document(
            document_id=document_id,
            filename=filename,
            extracted_text=extracted_text,
            word_index=word_index,
            page_count=page_count
        )
        
        if result:
            log(f"✅ Successfully uploaded: {document_id}")
            return True
        else:
            log(f"❌ Failed to upload: {document_id}")
            return False
            
    except Exception as e:
        log(f"❌ Error uploading {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_documents():
    """List all documents in the database"""
    try:
        client = SupabaseClient()
        docs = client.list_documents()
        
        if not docs:
            print("📭 No documents in database")
            return
        
        print(f"\n📚 Documents in Database ({len(docs)} total):")
        print("-" * 60)
        for doc in docs:
            print(f"  📄 {doc['document_id']}")
            print(f"     Filename: {doc['filename']}")
            print(f"     Pages: {doc['page_count']}")
            print(f"     Uploaded: {doc['created_at']}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")


def delete_document(document_id: str):
    """Delete a document from the database"""
    try:
        client = SupabaseClient()
        if client.delete_document(document_id):
            print(f"✅ Deleted: {document_id}")
        else:
            print(f"❌ Document not found: {document_id}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Upload reference documents to Supabase")
    parser.add_argument("files", nargs="*", help="PDF files to upload")
    parser.add_argument("--list", "-l", action="store_true", help="List all documents in database")
    parser.add_argument("--delete", "-d", metavar="DOC_ID", help="Delete a document by ID")
    parser.add_argument("--id", metavar="DOC_ID", help="Custom document ID (only for single file)")
    
    args = parser.parse_args()
    
    if args.list:
        list_documents()
        return
    
    if args.delete:
        delete_document(args.delete)
        return
    
    if not args.files:
        parser.print_help()
        print("\n📌 Examples:")
        print('  python upload_reference.py "path/to/document.pdf"')
        print('  python upload_reference.py file1.pdf file2.pdf')
        print('  python upload_reference.py --list')
        print('  python upload_reference.py --delete P6100283.1')
        return
    
    # Upload files
    success_count = 0
    for filepath in args.files:
        doc_id = args.id if len(args.files) == 1 else None
        if upload_document(filepath, doc_id):
            success_count += 1
    
    print(f"\n📊 Summary: {success_count}/{len(args.files)} files uploaded successfully")


if __name__ == "__main__":
    main()
