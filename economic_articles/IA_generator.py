
import pandas as pd
import random
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
import time
import os
import re
from typing import List, Dict, Optional

# file paths
INPUT_CSV = "clean_economic_dataset.csv"
OUTPUT_CSV = "ia_generated_dataset.csv"

#API Configuration
API_KEY = "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"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_PARALLEL_REQUESTS = 25
TIMEOUT = 20
RETRIES = 2

# Generation settings 
NUM_TEXTS_TO_GENERATE = 300
MIN_WORDS = 230
MODEL_NAME = "llama-3.1-8b-instant"


class TextGenerator:
    """Handles communication with the API and text generation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def __aenter__(self):
        """Initialize async session with limits and timeouts."""
        timeout = aiohttp.ClientTimeout(total=TIMEOUT, connect=5)
        connector = aiohttp.TCPConnector(
            limit=MAX_PARALLEL_REQUESTS * 3,
            limit_per_host=MAX_PARALLEL_REQUESTS * 2,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=self.headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async session."""
        if self.session:
            await self.session.close()

    def build_prompt(self, topic: str, is_speech: bool, language: str) -> List[Dict]:
        """Create a prompt based on language and style."""
        if language == "French":
            if is_speech:
                prompt = f"Discours économique de 280 mots sur: {topic}. Style oral naturel, pas de formatage."
            else:
                prompt = f"Article économique de 280 mots sur: {topic}. Style journalistique naturel, pas de formatage."
        else:
            if is_speech:
                prompt = f"Professional economic speech of 280 words about: {topic}. Natural oral style, no formatting."
            else:
                prompt = f"Economic article of 280 words about: {topic}. Natural journalistic style, no formatting."
        return [{"role": "user", "content": prompt}]

    def clean_text(self, text: str) -> Optional[str]:
        """Basic cleanup and filtering of generated text."""
        if not text or len(text) < 800:
            return None

        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'^[-•]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text.split()) < MIN_WORDS:
            return None

        return text

    async def generate_one_text(self, topic: str, is_speech: bool, language: str) -> Optional[str]:
        """Generate a single text with retries."""
        messages = self.build_prompt(topic, is_speech, language)
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 400,
            "temperature": 0.8,
            "stream": False
        }

        try:
            async with self.session.post(API_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    raw_text = data["choices"][0]["message"]["content"]
                    return self.clean_text(raw_text)
                elif response.status == 429:
                    await asyncio.sleep(0.2)
                    return await self.generate_one_text(topic, is_speech, language)
                else:
                    return None
        except:
            return None

    async def generate_batch(self, topics: List[tuple]) -> List[Dict]:
        """Generate many texts in parallel."""
        semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

        async def generate(config):
            topic, is_speech, language = config
            async with semaphore:
                text = await self.generate_one_text(topic, is_speech, language)
                if text:
                    return {
                        "text": text,
                        "source": "AI",
                        "is_speech": is_speech,
                        "language": language,
                        "label": 1
                    }
                return None

        print(f"Generating {len(topics)} texts...")
        results = await tqdm.gather(
            *[generate(config) for config in topics],
            desc="Text Generation"
        )

        valid = [r for r in results if r is not None]
        success_rate = len(valid) / len(topics) * 100
        print(f"{len(valid)}/{len(topics)} successful ({success_rate:.1f}%)")

        return valid


def prepare_topics(input_csv: str) -> List[tuple]:
    """Prepare a large list of (topic, is_speech, language) tuples."""
    print("Preparing topics...")
    try:
        df_real = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"File {input_csv} not found")
        return []

    needed = 1000
    sampled = df_real.sample(n=needed, replace=True) if len(df_real) < needed else df_real.sample(n=needed)

    languages = ["French", "English"]
    topics = []

    for _, row in sampled.iterrows():
        topic = row.get("summary", "").strip() or str(row.get("text", "Economic analysis"))[:200]
        language = random.choice(languages)
        is_speech = random.choice([True, False])
        topics.append((topic, is_speech, language))

    print(f"{len(topics)} topics ready")
    return topics


async def main():
    """Main async workflow."""
    start = time.time()
    print(f"Generating {NUM_TEXTS_TO_GENERATE} texts using {MODEL_NAME}")

    topics = prepare_topics(INPUT_CSV)
    if not topics:
        return

    async with TextGenerator(API_KEY) as generator:
        results = await generator.generate_batch(topics)

    final_results = results[:NUM_TEXTS_TO_GENERATE]

    if final_results:
        pd.DataFrame(final_results).to_csv(OUTPUT_CSV, index=False, sep=";", encoding="utf-8")

        duration = time.time() - start
        avg_words = sum(len(r['text'].split()) for r in final_results) / len(final_results)
        print(f"\n Done in {duration/60:.1f} min")
        print(f" Average words: {avg_words:.0f}")
        print(f"Saved to {OUTPUT_CSV}")
    else:
        print("No text generated")


if __name__ == "__main__":
    asyncio.run(main())
