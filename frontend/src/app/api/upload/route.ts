import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import pdfParse from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

export const maxDuration = 60; // Max allowed for Vercel Hobby on some plans, safely 10s default

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;
    const userId = formData.get("userId") as string; // Temporary until full auth

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }
    
    if (!userId) {
      return NextResponse.json({ error: "User ID required" }, { status: 400 });
    }

    if (file.size > 5 * 1024 * 1024) {
      return NextResponse.json({ error: "File exceeds 5MB limit" }, { status: 400 });
    }

    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
    
    if (!supabaseUrl || !supabaseServiceKey) {
      return NextResponse.json({ error: "Supabase config missing" }, { status: 500 });
    }

    // Use service role key to perform backend operations
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // 1. Upload to Supabase Storage
    const fileName = `${userId}/${Date.now()}-${file.name.replace(/[^a-zA-Z0-9.-]/g, "_")}`;
    const arrayBuffer = await file.arrayBuffer();
    
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from("documents")
      .upload(fileName, arrayBuffer, {
        contentType: file.type,
      });

    if (uploadError) {
      return NextResponse.json({ error: "Failed to upload file to storage: " + uploadError.message }, { status: 500 });
    }

    // 2. Insert Document Record
    const { data: docRecord, error: docError } = await supabase
      .from("documents")
      .insert({
        user_id: userId,
        title: file.name,
        storage_path: uploadData.path,
        status: "processing"
      })
      .select()
      .single();

    if (docError) {
      return NextResponse.json({ error: "Failed to insert document record" }, { status: 500 });
    }

    // 3. Extract Text (assuming PDF for now)
    let text = "";
    if (file.name.endsWith(".pdf")) {
      const buffer = Buffer.from(arrayBuffer);
      const pdfData = await pdfParse(buffer);
      text = pdfData.text;
    } else {
      // Very naive text extraction for plain text files
      const textDecoder = new TextDecoder("utf-8");
      text = textDecoder.decode(arrayBuffer);
    }

    if (!text.trim()) {
      await supabase.from("documents").update({ status: "failed", error_message: "No text found in file" }).eq("id", docRecord.id);
      return NextResponse.json({ error: "No text found in file" }, { status: 400 });
    }

    // 4. Chunk Text
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 800,
      chunkOverlap: 100,
    });
    const chunks = await splitter.createDocuments([text]);

    // 5. Insert Chunks (without embeddings initially)
    const chunkRecords = chunks.map((chunk, index) => ({
      document_id: docRecord.id,
      user_id: userId,
      content: chunk.pageContent,
      page_number: 1, // pdf-parse doesn't easily give page-by-page mapping, simplified for now
      chunk_index: index,
      // embedding is null initially
    }));

    const { error: chunkInsertError } = await supabase
      .from("document_chunks")
      .insert(chunkRecords);

    if (chunkInsertError) {
      await supabase.from("documents").update({ status: "failed", error_message: "Failed to insert chunks" }).eq("id", docRecord.id);
      return NextResponse.json({ error: "Failed to insert chunks" }, { status: 500 });
    }

    return NextResponse.json({ 
      message: "Document uploaded and chunked. Ready for embedding.",
      documentId: docRecord.id 
    });

  } catch (error: any) {
    console.error("Upload error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
