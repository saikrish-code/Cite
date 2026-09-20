import asyncio
import httpx
import json

async def test_llm():
    url = "http://localhost:8000/api/v1/chat"
    headers = {"Content-Type": "application/json"}
    # Note: testing without auth might fail if endpoints are protected.
    # Let's try directly against the LLM service to bypass auth for a quick test.
    print("Testing backend API...")

if __name__ == "__main__":
    asyncio.run(test_llm())
