-- Supabase Table Schema for PDF Compare
-- Run this in Supabase SQL Editor (Dashboard > SQL Editor > New Query)

-- Create the documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    extracted_text TEXT,
    word_index JSONB,
    page_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations for authenticated users
-- For development, allow anonymous access
CREATE POLICY "Allow anonymous access" ON documents
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Add comment to table
COMMENT ON TABLE documents IS 'Stores reference PDF documents for comparison';
COMMENT ON COLUMN documents.document_id IS 'Unique identifier extracted from filename (e.g., P6100283.1)';
COMMENT ON COLUMN documents.extracted_text IS 'Full OCR text for LLM comparison';
COMMENT ON COLUMN documents.word_index IS 'Word positions JSON for visual highlighting';
