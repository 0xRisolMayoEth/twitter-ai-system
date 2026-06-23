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

def write_tweet(topic, angle, hashtags):
    prompt = load_prompt("writer.md")
    prompt = prompt.replace("{topic}", topic).replace("{angle}", angle).replace("{hashtags}", " ".join(hashtags))
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    tweet = response.choices[0].message.content.strip()
    print(f"✍️ Tweet draft: {tweet}")
    return tweet

if __name__ == "__main__":
    print(write_tweet("retro gaming SNES vs Genesis", "nostalgia debate seru", ["#RetroGaming", "#SNES"]))
