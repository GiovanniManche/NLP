import json
import pandas as pd
import re
from pathlib import Path

# === Paramètres ===
data_folder = "economic_articles"
database_filename = "database_economic_articles.json"
output_csv = "clean_economic_dataset.csv"
ia_generated_filename = "IA_generated_dataset.csv"  # Fichier IA dans le dossier

# === Chargement du fichier JSON ===
database_path = Path(data_folder) / database_filename
with open(database_path, "r", encoding="utf-8") as f:
    data = json.load(f)

articles = data.get("articles", [])
print(f"Articles chargés : {len(articles)}")

# === Fonction de nettoyage du texte ===
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)        # supprimer les URLs
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", text)  # dates type 12/05/2024
    text = re.sub(r"\d{4}", " ", text)                   # années isolées
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)          # caractères spéciaux
    text = re.sub(r"\s+", " ", text)                     # espaces multiples
    return text.strip()

# === Création du DataFrame nettoyé ===
df = pd.DataFrame([
    {
        "text": clean_text(article.get("content", "")),
        "source": article.get("source", "Unknown"),
        "content_type": article.get("content_type", "Economic Article"),
        "is_likely_speech": article.get("is_likely_speech", False),
        "language": article.get("language", "English"),
        "label": 0,
    }
    for article in articles if len(article.get("content", "")) > 50
])

print(f"Articles après filtrage : {len(df)}")

# === Merge avec IA_generated_dataset.csv s'il existe ===
ia_path = Path(data_folder) / ia_generated_filename
if ia_path.exists():
    df_ia = pd.read_csv(ia_path, sep=";")  # Lecture avec ; car CSV IA déjà en ;
    print(f"Articles IA chargés : {len(df_ia)}")
    
    # Merge en priorisant les données IA si déjà présentes
    df_merged = pd.concat([df_ia, df], ignore_index=True).drop_duplicates(subset=["text"], keep="first")
else:
    print("⚠️ IA_generated_dataset.csv non trouvé. On enregistre uniquement le nouveau dataset.")
    df_merged = df

# === Sauvegarde du CSV final avec séparateur `;` ===
output_path = Path(data_folder) / output_csv
df_merged.to_csv(output_path, index=False, encoding="utf-8", sep=";")
print(f"✅ Dataset fusionné sauvegardé dans {output_path}")
print(df_merged.head())
