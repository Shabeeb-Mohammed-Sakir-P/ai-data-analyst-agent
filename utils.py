import json
import re


def parse_json_response(raw_response: str) -> list:
    """
    LLMs sometimes wrap JSON in ```json code fences or add extra text
    despite instructions not to. This function cleans that up before parsing.
    Shared across all agents that need structured JSON output from the LLM.
    """
    cleaned = raw_response.strip()

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("Warning: Could not parse LLM response as JSON. Raw response was:")
        print(raw_response)
        return []