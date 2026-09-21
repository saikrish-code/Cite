# Prompt templates for LLM-as-a-judge evaluation

FAITHFULNESS_PROMPT = """You are an impartial judge evaluating the faithfulness of an AI-generated answer.
Your task is to determine if ALL the claims made in the ANSWER are directly supported by the provided CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Evaluate step-by-step:
1. Extract all factual claims from the ANSWER.
2. For each claim, check if it can be directly inferred from the CONTEXT.
3. If ANY claim is not supported by or contradicts the CONTEXT, faithfulness is 0.
4. If ALL claims are supported, faithfulness is 1.

Return your final decision as a JSON object:
{{
  "reasoning": "your step-by-step reasoning",
  "score": 1 or 0
}}
"""

ANSWER_RELEVANCE_PROMPT = """You are an impartial judge evaluating how relevant an AI-generated answer is to the user's question.
The answer should directly address the core of the question without excessive irrelevant information.

QUESTION:
{question}

ANSWER:
{answer}

Evaluate step-by-step:
1. Identify the core intent of the QUESTION.
2. Check if the ANSWER addresses this intent.
3. Score from 0 to 1 based on relevance (1 = perfectly relevant and concise, 0.5 = partially relevant or too much extra info, 0 = irrelevant).

Return your final decision as a JSON object:
{{
  "reasoning": "your step-by-step reasoning",
  "score": float between 0 and 1
}}
"""

ANSWER_CORRECTNESS_PROMPT = """You are an impartial judge evaluating the correctness of an AI-generated answer compared to a ground truth answer.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

GENERATED ANSWER:
{answer}

Evaluate step-by-step:
1. Compare the factual content of the GENERATED ANSWER against the GROUND TRUTH.
2. If the GENERATED ANSWER contains all the key facts from the GROUND TRUTH and is correct, score 1.
3. If it is partially correct or misses key facts, score 0.5.
4. If it is completely wrong or contradicts the GROUND TRUTH, score 0.

Return your final decision as a JSON object:
{{
  "reasoning": "your step-by-step reasoning",
  "score": float between 0 and 1
}}
"""

CITATION_ENTAILMENT_PROMPT = """You are an impartial judge evaluating if a specific cited snippet entails the sentence that cites it.

SENTENCE WITH CITATION:
{sentence}

CITED SNIPPET:
{snippet}

Evaluate step-by-step:
1. Read the SENTENCE and understand its claim.
2. Read the CITED SNIPPET.
3. Determine if the CITED SNIPPET provides direct evidence for the claim in the SENTENCE.
4. If it provides direct evidence, score 1. Otherwise, score 0.

Return your final decision as a JSON object:
{{
  "reasoning": "your step-by-step reasoning",
  "score": 1 or 0
}}
"""
