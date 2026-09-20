import { NextRequest } from "next/server";
import { createClient } from "@supabase/supabase-js";
import { GoogleGenerativeAI } from "@google/generative-ai";

export const maxDuration = 60; // Max allowed for Vercel Hobby

export async function POST(req: NextRequest) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      function sendEvent(event: string, data: any) {
        controller.enqueue(encoder.encode(`event: ${event}\n`));
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
      }

      try {
        const body = await req.json();
        const { query, document_id, user_id } = body;
        
        // Use user_id from body for now (Stage 4 will use auth headers)
        const activeUserId = user_id;

        if (!query || !activeUserId) {
          sendEvent("error", { detail: "Query and User ID are required." });
          controller.close();
          return;
        }

        const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
        const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
        const geminiApiKey = process.env.GEMINI_API_KEY;

        if (!supabaseUrl || !supabaseServiceKey || !geminiApiKey) {
          sendEvent("error", { detail: "Config missing" });
          controller.close();
          return;
        }

        const supabase = createClient(supabaseUrl, supabaseServiceKey);
        const genAI = new GoogleGenerativeAI(geminiApiKey);

        // 1. Embed Query
        const embedModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
        const queryEmbeddingRes = await embedModel.embedContent(query);
        const queryEmbedding = queryEmbeddingRes.embedding.values;

        // 2. Hybrid Search (RRF)
        // Call the SQL function we created in Stage 1
        const { data: searchResults, error: searchError } = await supabase.rpc(
          "hybrid_search",
          {
            query_text: query,
            query_embedding: queryEmbedding,
            match_count: 5,
            full_text_weight: 1.0,
            semantic_weight: 1.0,
            rrf_k: 60
          }
        );

        if (searchError) {
          throw new Error("Search failed: " + searchError.message);
        }

        // 3. Grounded Prompt
        let contextFound = false;
        let citations: any[] = [];
        let systemPrompt = "";

        if (!searchResults || searchResults.length === 0) {
          systemPrompt = "You are a helpful assistant. Keep your answer concise and conversational. Do not make up facts.";
          contextFound = false;
        } else {
          contextFound = true;
          let contextString = "";
          searchResults.forEach((res: any, idx: number) => {
            const citeId = idx + 1;
            contextString += `[${citeId}] (Page ${res.page_number}): ${res.content}\n\n`;
            citations.push({
              source_id: citeId,
              filename: "Document", // Simplification
              page: res.page_number,
              snippet: res.content.substring(0, 200) + "...",
              score: res.similarity
            });
          });

          systemPrompt = `You are a RAG assistant answering strictly based on the provided context.
If the context does not contain the answer, say "I don't know based on the provided context."
Cite sources inline as [1], [2], etc.

Context:
${contextString}
`;
        }

        // 4. Stream LLM Response
        const chatModel = genAI.getGenerativeModel({ 
          model: "gemini-1.5-flash",
          systemInstruction: systemPrompt 
        });

        const result = await chatModel.generateContentStream(query);
        let fullAnswer = "";

        for await (const chunk of result.stream) {
          const chunkText = chunk.text();
          fullAnswer += chunkText;
          sendEvent("token", { token: chunkText });
        }

        // 5. Send Citations and Done
        sendEvent("citations", {
          citations: contextFound ? citations : [],
          context_found: contextFound,
          retrieval_metrics: {
            mode: "hybrid",
            total_retrieval_ms: 0 // Mock for now
          }
        });

        sendEvent("done", {
          query,
          answer: fullAnswer
        });

        // (Stage 4 will persist to chat_messages)

      } catch (error: any) {
        console.error("Chat error:", error);
        sendEvent("error", { detail: error.message });
      } finally {
        controller.close();
      }
    }
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive"
    }
  });
}
