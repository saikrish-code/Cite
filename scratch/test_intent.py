import asyncio
import os
import sys

# Add backend directory to sys.path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.schemas.chat import ChatHistoryEntry
from app.services.intent import IntentClassifier, IntentCategory

async def test_intent_classifier():
    print("Testing Intent Classifier...")
    classifier = IntentClassifier()

    print("\n--- Test 1: CONVERSATIONAL ---")
    intent = await classifier.aclassify("Thanks for your help!")
    print(f"Query: 'Thanks for your help!' -> Intent: {intent}")
    assert intent == IntentCategory.CONVERSATIONAL

    print("\n--- Test 2: RESEARCH ---")
    intent = await classifier.aclassify("What dataset did they use for training?")
    print(f"Query: 'What dataset did they use for training?' -> Intent: {intent}")
    assert intent == IntentCategory.RESEARCH

    print("\n--- Test 3: AMBIGUOUS ---")
    intent = await classifier.aclassify("Tell me more about it.")
    print(f"Query: 'Tell me more about it.' -> Intent: {intent}")
    assert intent == IntentCategory.AMBIGUOUS

    print("\n--- Test 4: History Resolution (RESEARCH) ---")
    history = [
        ChatHistoryEntry(
            id="1", query="What is RAG?", answer="RAG is Retrieval-Augmented Generation.",
            citations=[], context_found=True, created_at="2023-01-01"
        )
    ]
    intent = await classifier.aclassify("Can you explain that in more detail?", history=history)
    print(f"Query: 'Can you explain that in more detail?' (with history about RAG) -> Intent: {intent}")
    assert intent == IntentCategory.RESEARCH

    print("\nAll tests passed locally (if no assertions failed)!")

if __name__ == "__main__":
    asyncio.run(test_intent_classifier())
