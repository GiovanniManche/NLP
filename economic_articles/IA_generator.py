import pandas as pd
import random
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
import time
import os
import re
from typing import List, Dict, Optional

input_csv = "clean_economic_dataset.csv"
output_csv = "ia_generated_dataset.csv"

# Configuration ULTRA-RAPIDE
GROQ_API_KEY = "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"
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_CONCURRENT_REQUESTS = 25  # BEAUCOUP plus agressif
TIMEOUT_SECONDS = 20  # Plus court
RETRY_ATTEMPTS = 2  # Moins de retry

TARGET_TEXTS = 300
MIN_WORDS = 230  # Un peu plus souple
GROQ_MODEL = "llama-3.1-8b-instant"  # LE PLUS RAPIDE

class UltraFastGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    async def __aenter__(self):
        # Configuration ultra-agressive
        timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS, connect=5)
        connector = aiohttp.TCPConnector(
            limit=MAX_CONCURRENT_REQUESTS * 3,
            limit_per_host=MAX_CONCURRENT_REQUESTS * 2,
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
        if self.session:
            await self.session.close()

    def create_fast_prompt(self, theme: str, is_speech: bool, language: str) -> List[Dict]:
        """Prompts courts et directs pour la vitesse"""
        
        if language == "French":
            if is_speech:
                prompt = f"Discours économique professionnel de 280 mots sur: {theme}. Style oral naturel, pas de formatage."
            else:
                prompt = f"Article économique de 280 mots sur: {theme}. Style journalistique naturel, pas de formatage."
        else:
            if is_speech:
                prompt = f"Professional economic speech of 280 words about: {theme}. Natural oral style, no formatting."
            else:
                prompt = f"Economic article of 280 words about: {theme}. Natural journalistic style, no formatting."

        return [{"role": "user", "content": prompt}]

    def quick_clean(self, text: str) -> Optional[str]:
        """Nettoyage rapide et efficace"""
        if not text or len(text) < 800:  # Approximation rapide
            return None
            
        # Nettoyage ultra-rapide
        text = re.sub(r'\*+([^*]+)\*+', r'\1', text)  # Astérisques
        text = re.sub(r'#{1,6}\s*', '', text)  # Headers
        text = re.sub(r'^[-•]\s*', '', text, flags=re.MULTILINE)  # Listes
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)  # Numéros
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Vérification rapide des mots
        if len(text.split()) < MIN_WORDS:
            return None
            
        return text

    async def generate_fast(self, theme: str, is_speech: bool, language: str) -> Optional[str]:
        """Génération ultra-rapide avec minimum de vérifications"""
        messages = self.create_fast_prompt(theme, is_speech, language)
        
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "max_tokens": 400,  # Juste ce qu'il faut
            "temperature": 0.8,
            "stream": False
        }

        try:
            async with self.session.post(GROQ_BASE_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    raw_text = data["choices"][0]["message"]["content"]
                    return self.quick_clean(raw_text)
                elif response.status == 429:
                    await asyncio.sleep(0.2)  # Attente très courte
                    return await self.generate_fast(theme, is_speech, language)
                else:
                    return None
        except:
            return None

    async def mass_generate(self, themes_configs: List[tuple]) -> List[Dict]:
        """Génération de masse ultra-rapide"""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def generate_one(config):
            theme, is_speech, language = config
            async with semaphore:
                text = await self.generate_fast(theme, is_speech, language)
                if text:
                    return {
                        "text": text,
                        "source": "IA",
                        "is_likely_speech": is_speech,
                        "language": language,
                        "label": 1
                    }
                return None
        
        print(f"🚀 GÉNÉRATION ULTRA-RAPIDE: {len(themes_configs)} textes...")
        results = await tqdm.gather(
            *[generate_one(config) for config in themes_configs],
            desc="Ultra-Fast Gen"
        )
        
        valid = [r for r in results if r is not None]
        success_rate = len(valid) / len(themes_configs) * 100
        print(f"⚡ {len(valid)}/{len(themes_configs)} réussis ({success_rate:.1f}%)")
        
        return valid

def prepare_massive_themes(input_csv: str) -> List[tuple]:
    """Prépare BEAUCOUP de thèmes d'un coup"""
    print("📖 Préparation massive des thèmes...")
    
    try:
        df_real = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"❌ Fichier {input_csv} non trouvé")
        return []
    
    # Générer 500 thèmes d'un coup (pour avoir 300+ garantis)
    needed = 1000
    if len(df_real) < needed:
        sampled = df_real.sample(n=needed, replace=True)
    else:
        sampled = df_real.sample(n=needed)
    
    configs = []
    languages = ["French", "English"]
    
    for _, row in sampled.iterrows():
        # Thème
        if isinstance(row.get("summary", ""), str) and row["summary"].strip():
            theme = row["summary"]
        else:
            theme = str(row["text"])[:200] if pd.notna(row.get("text")) else "Economic analysis"
        
        # Configuration aléatoire
        language = random.choice(languages)
        is_speech = random.choice([True, False])
        
        configs.append((theme, is_speech, language))
    
    print(f"✅ {len(configs)} configurations prêtes")
    return configs

def setup_api():
    """Setup API rapide"""
    global GROQ_API_KEY
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        GROQ_API_KEY = env_key
        return True
    
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print("❌ Définissez GROQ_API_KEY")
        return False
    return True

async def ultra_fast_main():
    """Main ultra-rapide"""
    if not setup_api():
        return
    
    start = time.time()
    print(f"🎯 GÉNÉRATION ULTRA-RAPIDE DE {TARGET_TEXTS} TEXTES")
    print(f"🚀 Modèle: {GROQ_MODEL} | Concurrence: {MAX_CONCURRENT_REQUESTS}")
    
    # Préparation
    configs = prepare_massive_themes(input_csv)
    if not configs:
        return
    
    # Génération MASSIVE
    async with UltraFastGenerator(GROQ_API_KEY) as generator:
        all_results = await generator.mass_generate(configs)
    
    # Prendre les 300 premiers (ou tous si moins)
    final_results = all_results[:TARGET_TEXTS]
    
    if final_results:
        # Sauvegarde immédiate avec séparateur point-virgule
        df = pd.DataFrame(final_results)
        df.to_csv(output_csv, index=False, encoding="utf-8", sep=";")
        
        # Stats rapides
        duration = time.time() - start
        speed = len(final_results) * 3600 / duration
        avg_words = sum(len(r['text'].split()) for r in final_results) / len(final_results)
        
        fr_count = sum(1 for r in final_results if r['language'] == 'French')
        
        print(f"\n🎉 TERMINÉ EN {duration/60:.1f} MINUTES!")
        print(f"📝 {len(final_results)} textes générés")
        print(f"⚡ {speed:.0f} textes/heure")
        print(f"📊 FR: {fr_count} | EN: {len(final_results)-fr_count}")
        print(f"📏 Moyenne: {avg_words:.0f} mots")
        print(f"💾 {output_csv}")
        
    else:
        print("❌ Échec de génération")

if __name__ == "__main__":
    # Installation rapide si nécessaire
    try:
        import aiohttp
        from tqdm.asyncio import tqdm
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "aiohttp", "tqdm"])
    
    asyncio.run(ultra_fast_main())