import os
from dotenv import load_dotenv
from groq import Groq

# Loads the values from your .env file into the environment
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "Say hello in exactly 5 words."}
    ]
)

print(response.choices[0].message.content)