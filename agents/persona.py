import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
MODEL = os.getenv("AI_MODEL", "claude-sonnet-4-6")
PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

def load_prompt(filename):
    with open(os.path.join(PROMPT_PATH, filename), "r") as f:
        return f.read()

def apply_persona(tweet):
    prompt = load_prompt("persona.md").replace("{tweet}", tweet)
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content.strip()
    print(f"🎭 Persona: {result}")
    return result

if __name__ == "__main__":
    print(apply_persona("SNES vs Genesis adalah debat yang tidak ada habisnya."))
