import json
import re


def parse_json_response(raw_response: str) -> list:
    """
    LLMs sometimes wrap JSON in ```json code fences, add extra text, or
    include stray trailing characters despite instructions not to.
    This function cleans that up before parsing.
    """
    cleaned = raw_response.strip()

    # Remove markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Common LLM mistake: extra characters after the array/object closes.
        # Try trimming to the last valid ] or } and parsing again.
        for closer in ["]", "}"]:
            last_index = cleaned.rfind(closer)
            if last_index != -1:
                trimmed = cleaned[: last_index + 1]
                try:
                    return json.loads(trimmed)
                except json.JSONDecodeError:
                    continue

        print("Warning: Could not parse LLM response as JSON. Raw response was:")
        print(raw_response)
        return []