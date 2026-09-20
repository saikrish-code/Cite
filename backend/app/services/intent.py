"""Intent classification layer for the RAG pipeline."""

import json
import logging
import re
from enum import Enum
from typing import Any

from app.schemas.chat import ChatHistoryEntry
from app.services.llm import BaseLLMClient, get_llm_client

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    CONVERSATIONAL = "CONVERSATIONAL"
    RESEARCH = "RESEARCH"
    AMBIGUOUS = "AMBIGUOUS"


INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for a research-paper RAG assistant.

Classify the user's latest message into exactly one category:

CONVERSATIONAL:
The user is greeting, thanking, acknowledging, confirming, saying goodbye, making casual conversation, or otherwise interacting socially without requesting information from the research documents.

RESEARCH:
The user is requesting factual, academic, scientific, technical, or research-related information that should be answered using the indexed research papers/documents.

AMBIGUOUS:
The user's intent is unclear and cannot confidently be classified as conversational or research-related.

Use conversation history when available to resolve references such as 'this', 'that', 'why', or 'tell me more'.

Return ONLY valid JSON in this format:
{
  "intent": "CONVERSATIONAL"
}

Do not answer the user's question. Only classify the intent. Do not output markdown code blocks, just raw JSON."""


class IntentClassifier:
    """Classifies user query intent to determine if RAG retrieval is necessary."""

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client or get_llm_client()

    def _build_prompt(
        self, query: str, history: list[ChatHistoryEntry] | None = None
    ) -> str:
        prompt = ""
        if history and len(history) > 0:
            prompt += "Conversation History:\n"
            # Sort history by created_at ascending (oldest first) if provided as ChatHistoryEntry
            # or assume it's already in chronological order depending on how it's fetched.
            # We'll expect the caller to pass it in chronological order.
            for entry in history:
                prompt += f"User: {entry.query}\nAssistant: {entry.answer}\n\n"

        prompt += f"Latest User Message:\n{query}\n"
        return prompt

    def _parse_response(self, raw_response: str) -> IntentCategory:
        """Parse LLM JSON response into IntentCategory, defaulting to RESEARCH on failure."""
        try:
            # Strip markdown json blocks if the model ignored instructions
            cleaned = re.sub(r"```json\s*", "", raw_response, flags=re.IGNORECASE)
            cleaned = re.sub(r"```\s*", "", cleaned)
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            intent_str = data.get("intent", "").upper()
            
            if intent_str in [i.value for i in IntentCategory]:
                return IntentCategory(intent_str)
            
            logger.warning("Unrecognized intent '%s', defaulting to RESEARCH.", intent_str)
            return IntentCategory.RESEARCH
            
        except Exception as e:
            logger.warning("Failed to parse intent JSON: %s. Response was: %s", e, raw_response)
            return IntentCategory.RESEARCH

    def classify(
        self, query: str, history: list[ChatHistoryEntry] | None = None
    ) -> IntentCategory:
        """Synchronously classify intent."""
        user_prompt = self._build_prompt(query, history)
        try:
            response = self.llm_client.generate(
                prompt=user_prompt, system_prompt=INTENT_CLASSIFIER_PROMPT
            )
            return self._parse_response(response)
        except Exception as e:
            logger.error("Intent classification API failed: %s", e)
            return IntentCategory.RESEARCH

    async def aclassify(
        self, query: str, history: list[ChatHistoryEntry] | None = None
    ) -> IntentCategory:
        """Asynchronously classify intent."""
        user_prompt = self._build_prompt(query, history)
        try:
            response = await self.llm_client.agenerate(
                prompt=user_prompt, system_prompt=INTENT_CLASSIFIER_PROMPT
            )
            return self._parse_response(response)
        except Exception as e:
            logger.error("Intent classification API failed: %s", e)
            return IntentCategory.RESEARCH
