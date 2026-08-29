import os
from dotenv import load_dotenv
from groq import Groq
from google import genai

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def call_llm(prompt: str) -> str:
    """
    Sends a prompt to an LLM and returns its text response.
    Tries Groq first (fast, free). If that fails for any reason,
    automatically falls back to Gemini instead.
    """
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    except Exception as groq_error:
        print(f"Groq failed ({groq_error}), falling back to Gemini...")

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
    result = call_llm("Say hello in exactly 5 words.")
    print(result)