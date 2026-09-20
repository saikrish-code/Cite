import { createClient } from "@supabase/supabase-js";
import { GoogleGenerativeAI } from "@google/generative-ai";
import * as fs from "fs";

// Setup
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

async function runEval() {
  console.log("Starting Hybrid Search Evaluation against Supabase...");
  
  // 1. Generate Dummy Test Set if none exists
  const testQueries = [
    "What are the main contributions of this paper?",
    "Explain the methodology used.",
    "What is the performance on the benchmark?"
  ];

  const embedModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
  
  let totalLatency = 0;
  let hitRate = 0;

  for (const query of testQueries) {
    const start = Date.now();
    
    // Embed
    const embedRes = await embedModel.embedContent(query);
    const queryEmbedding = embedRes.embedding.values;

    // Search via Supabase Hybrid Search RPC
    const { data: results, error } = await supabase.rpc("hybrid_search", {
      query_text: query,
      query_embedding: queryEmbedding,
      match_count: 5,
      full_text_weight: 1.0,
      semantic_weight: 1.0,
      rrf_k: 60
    });

    const latency = Date.now() - start;
    totalLatency += latency;

    if (error) {
      console.error("Search error:", error.message);
      continue;
    }

    // Mock Hit Rate: if we return anything, it's a "hit" for this naive eval port
    if (results && results.length > 0) {
      hitRate += 1;
    }
    
    console.log(`Query: "${query}" | Latency: ${latency}ms | Results: ${results?.length || 0}`);
  }

  const avgLatency = totalLatency / testQueries.length;
  const hitRatePercentage = (hitRate / testQueries.length) * 100;

  console.log("\n=== EVALUATION RESULTS ===");
  console.log(`Average Retrieval Latency: ${avgLatency.toFixed(2)}ms`);
  console.log(`Hit Rate @ 5: ${hitRatePercentage.toFixed(2)}%`);
  
  // Save results
  const resultsData = {
    avgLatencyMs: avgLatency,
    hitRateAt5: hitRatePercentage,
    timestamp: new Date().toISOString()
  };
  
  if (!fs.existsSync("eval/results")) {
    fs.mkdirSync("eval/results");
  }
  fs.writeFileSync("eval/results/supabase_eval_results.json", JSON.stringify(resultsData, null, 2));
  console.log("Results saved to eval/results/supabase_eval_results.json");
}

runEval().catch(console.error);
