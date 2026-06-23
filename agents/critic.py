import sys, os, json
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

def review_tweet(tweet):
    prompt = load_prompt("critic.md").replace("{tweet}", tweet)
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.choices[0].message.content.strip()
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        result = json.loads(text[start:end])
    except:
        result = {"skor": 7, "layak_post": True, "masalah": [], "tweet_revisi": tweet}
    print(f"📊 Review - Skor: {result.get('skor')}/10")
    return result

if __name__ == "__main__":
    print(review_tweet("Test tweet gaming wkwk 🎮 #Gaming"))
