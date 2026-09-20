-- Enable the pgvector extension to work with embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- Set up storage for document files (Supabase Storage)
-- Note: Storage buckets are typically created via the Supabase UI or API,
-- but we can insert the bucket definition if needed. We'll assume the user
-- creates a bucket called 'documents'.
insert into storage.buckets (id, name, public) 
values ('documents', 'documents', false)
on conflict (id) do nothing;

-- 1. Documents Table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    storage_path TEXT NOT NULL, -- Path in Supabase Storage
    status TEXT NOT NULL DEFAULT 'processing', -- processing, ready, failed
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Document Chunks Table
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER NOT NULL,
    embedding VECTOR(768), -- Gemini uses 768 dimensions. Adjust if using Cohere (e.g., 1024) or OpenAI.
    fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- 3. Chat Messages Table
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations JSONB, -- Store array of citations if any
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
-- HNSW index for fast vector similarity search (requires pgvector 0.5+)
CREATE INDEX idx_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- GIN index for fast full-text search
CREATE INDEX idx_document_chunks_fts ON document_chunks USING GIN (fts);

-- Index for filtering by user and document quickly
CREATE INDEX idx_document_chunks_user_doc ON document_chunks (user_id, document_id);
CREATE INDEX idx_documents_user ON documents (user_id);
CREATE INDEX idx_chat_messages_user ON chat_messages (user_id);


-- Row Level Security (RLS)
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- Policies: Users can only select, insert, update, delete their own rows
CREATE POLICY "Users can manage their own documents" 
ON documents FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage their own chunks" 
ON document_chunks FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage their own chat messages" 
ON chat_messages FOR ALL USING (auth.uid() = user_id);

-- Storage Policies: Users can manage their own uploaded files
CREATE POLICY "Users can upload their own documents"
ON storage.objects FOR INSERT TO authenticated
WITH CHECK (bucket_id = 'documents' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can read their own documents"
ON storage.objects FOR SELECT TO authenticated
USING (bucket_id = 'documents' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can delete their own documents"
ON storage.objects FOR DELETE TO authenticated
USING (bucket_id = 'documents' AND auth.uid()::text = (storage.foldername(name))[1]);


-- Hybrid Search Function using Reciprocal Rank Fusion (RRF)
-- Combines vector similarity and full-text search rankings.
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding VECTOR(768),
    match_count INT DEFAULT 10,
    full_text_weight FLOAT DEFAULT 1.0,
    semantic_weight FLOAT DEFAULT 1.0,
    rrf_k INT DEFAULT 60
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    content TEXT,
    page_number INTEGER,
    similarity FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    user_id_param UUID := auth.uid();
BEGIN
    RETURN QUERY
    WITH semantic_search AS (
        SELECT
            dc.id,
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding) AS rank,
            1 - (dc.embedding <=> query_embedding) AS similarity
        FROM document_chunks dc
        WHERE dc.user_id = user_id_param
        ORDER BY dc.embedding <=> query_embedding
        LIMIT 100
    ),
    keyword_search AS (
        SELECT
            dc.id,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(dc.fts, websearch_to_tsquery('english', query_text)) DESC) AS rank
        FROM document_chunks dc
        WHERE dc.user_id = user_id_param 
          AND dc.fts @@ websearch_to_tsquery('english', query_text)
        ORDER BY ts_rank_cd(dc.fts, websearch_to_tsquery('english', query_text)) DESC
        LIMIT 100
    ),
    rrf AS (
        SELECT
            COALESCE(ss.id, ks.id) AS chunk_id,
            COALESCE(ss.similarity, 0.0) AS max_similarity,
            (
                COALESCE(semantic_weight / (ss.rank + rrf_k), 0.0) +
                COALESCE(full_text_weight / (ks.rank + rrf_k), 0.0)
            ) AS rrf_score
        FROM semantic_search ss
        FULL OUTER JOIN keyword_search ks ON ss.id = ks.id
    )
    SELECT
        dc.id,
        dc.document_id,
        dc.content,
        dc.page_number,
        r.max_similarity AS similarity
    FROM rrf r
    JOIN document_chunks dc ON dc.id = r.chunk_id
    ORDER BY r.rrf_score DESC
    LIMIT match_count;
END;
$$;
