import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import { GoogleGenerativeAI } from "@google/generative-ai";

export const maxDuration = 60; // Max allowed for Vercel Hobby

export async function POST(req: NextRequest) {
  try {
    const { documentId, userId } = await req.json();

    if (!documentId || !userId) {
      return NextResponse.json({ error: "Missing parameters" }, { status: 400 });
    }

    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
    const geminiApiKey = process.env.GEMINI_API_KEY;
    
    if (!supabaseUrl || !supabaseServiceKey || !geminiApiKey) {
      return NextResponse.json({ error: "Config missing" }, { status: 500 });
    }

    const supabase = createClient(supabaseUrl, supabaseServiceKey);
    const genAI = new GoogleGenerativeAI(geminiApiKey);

    // 1. Fetch chunks without embeddings for this document
    const { data: chunks, error: chunksError } = await supabase
      .from("document_chunks")
      .select("id, content")
      .eq("document_id", documentId)
      .is("embedding", null)
      .limit(10); // Batch size to avoid timeouts and rate limits

    if (chunksError) {
      return NextResponse.json({ error: "Failed to fetch chunks" }, { status: 500 });
    }

    if (chunks.length === 0) {
      // All chunks embedded, mark document as ready
      await supabase.from("documents").update({ status: "ready" }).eq("id", documentId);
      return NextResponse.json({ message: "Embedding complete", status: "ready" });
    }

    // 2. Embed the batch
    const model = genAI.getGenerativeModel({ model: "text-embedding-004" });
    
    for (const chunk of chunks) {
      const result = await model.embedContent(chunk.content);
      const embedding = result.embedding.values;

      // 3. Update chunk in DB
      const { error: updateError } = await supabase
        .from("document_chunks")
        .update({ embedding })
        .eq("id", chunk.id);

      if (updateError) {
        console.error("Failed to update embedding for chunk:", chunk.id, updateError);
        // Continue with others
      }
    }

    return NextResponse.json({ 
      message: `Embedded ${chunks.length} chunks.`,
      status: "processing",
      processedCount: chunks.length
    });

  } catch (error: any) {
    console.error("Embed error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
