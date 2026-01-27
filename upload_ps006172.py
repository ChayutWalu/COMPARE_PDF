# -*- coding: utf-8 -*-
"""
Quick upload script with hardcoded paths to avoid PowerShell encoding issues
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from upload_reference import upload_document

# Hardcoded path to avoid PowerShell Thai character encoding issues
file_path = r"c:\Users\chayu\Desktop\PDF Compare\COMPARE_PDF\PDF_test\OCR_10.11.2025\ไฟล์สแกนเอกสารประกันภัย\PS006172dc.pdf"

if os.path.exists(file_path):
    print(f"✅ File found: {file_path}")
    upload_document(file_path, "PS006172dc")
else:
    print(f"❌ File not found: {file_path}")
    # Try to list files in directory
    dir_path = os.path.dirname(file_path)
    if os.path.exists(dir_path):
        print(f"\nFiles in directory:")
        for f in os.listdir(dir_path):
            print(f"  - {f}")
    else:
        print(f"Directory not found: {dir_path}")
