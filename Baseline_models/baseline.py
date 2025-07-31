import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_extraction import text
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Data loading
data_path = Path("economic_articles") / "clean_economic_dataset.csv"
df = pd.read_csv(data_path, sep=";", encoding="latin1")
print(f"Number of rows: {len(df)}")
base_dir = Path("Baseline_models")

# French + English Stopwords 
french_stopwords = set("""
au aux avec ce ces dans de des du elle en et eux il je la le leur lui ma mais me même mes 
moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton 
tu un une vos votre vous c d j l à m n s t y été étée étées étés étant étante étants étantes suis 
es est sommes êtes sont serai seras sera serons serez seront serais serait serions seriez seraient 
étais était étions étiez étaient fus fut fûmes fûtes furent sois soit soyons soyez soient 
aie aies ait ayons ayez aient eu eus eut eûmes eûtes eurent aie ayant ayante ayantes ayants
alors ainsi après avant bien car comme comment donc où puis quand sans selon sous tout
très plus moins aussi encore jamais toujours déjà enfin surtout cependant pourtant néanmoins
toutefois ensuite puis finalement donc par conséquent en effet c'est-à-dire autrement dit
""".split())

stop_words = list(text.ENGLISH_STOP_WORDS.union(french_stopwords))

# Advanced linguistic feature extraction 
def extract_linguistic_features(text):
    """Extracts linguistic features specific to AI detection"""
    text = str(text)
    
    features = {}
    
    # Length and structure
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['sentence_count'] = len(re.findall(r'[.!?]+', text))
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
    
    # Punctuation and formatting
    features['punctuation_ratio'] = len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['comma_count'] = text.count(',')
    features['semicolon_count'] = text.count(';')
    features['colon_count'] = text.count(':')
    
    # Repetitions and lexical diversity
    words = text.lower().split()
    word_freq = Counter(words)
    features['unique_words_ratio'] = len(set(words)) / max(len(words), 1)
    features['most_common_word_freq'] = word_freq.most_common(1)[0][1] / max(len(words), 1) if words else 0
    
    # Typical AI patterns
    ai_patterns = [
        r'\ben effet\b', r'\bpar conséquent\b', r'\btoutefois\b', r'\bcependant\b',
        r'\bnéanmoins\b', r'\bainsi\b', r'\bpar ailleurs\b', r'\ben outre\b',
        r'\bde plus\b', r'\bnotamment\b', r'\bil convient de\b', r'\bil est important de\b',
        r'\bil faut noter\b', r'\bdans ce contexte\b', r'\ben conclusion\b'
    ]
    features['ai_pattern_count'] = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in ai_patterns)
    
    # Transition words and logical connectors
    transition_words = ['donc', 'ainsi', 'par conséquent', 'en effet', 'cependant', 'néanmoins', 
                       'toutefois', 'en outre', 'de plus', 'furthermore', 'however', 'therefore', 
                       'moreover', 'nevertheless', 'consequently']
    features['transition_words_count'] = sum(text.lower().count(word) for word in transition_words)
    
    # Syntactic complexity
    features['complex_punctuation'] = text.count(';') + text.count(':') + text.count('—')
    features['parentheses_count'] = text.count('(') + text.count(')')
    features['quotes_count'] = text.count('"') + text.count("'")
    
    # Text entropy (measure of predictability)
    if words:
        word_probs = np.array(list(word_freq.values())) / len(words)
        features['text_entropy'] = -np.sum(word_probs * np.log2(word_probs + 1e-10))
    else:
        features['text_entropy'] = 0
    
    return features

# Improved text cleaning 
def clean_text(text):
    text = str(text).lower()
    # Preserve some important punctuation for analysis
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:()\"'-]", " ", text)  # Keep more punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Feature extraction 
print("Extracting linguistic features...")
linguistic_features = []
df["text_clean"] = df["text"].apply(clean_text)

for text in df["text_clean"]:
    features = extract_linguistic_features(text)
    linguistic_features.append(features)

features_df = pd.DataFrame(linguistic_features)
print(f"Linguistic features extracted: {features_df.shape[1]} features")

#Exploratory analysis 
def plot_feature_analysis(features_df, labels):
    """Analysis of differences between AI and human texts"""
    features_df['label'] = labels
    
    # Top most discriminative features
    important_features = ['avg_sentence_length', 'unique_words_ratio', 'ai_pattern_count', 
                         'transition_words_count', 'text_entropy', 'punctuation_ratio']
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(important_features, 1):
        plt.subplot(2, 3, i)
        ai_data = features_df[features_df['label'] == 1][feature]
        human_data = features_df[features_df['label'] == 0][feature]
        
        plt.hist(ai_data, alpha=0.5, label='AI', bins=20, color='red')
        plt.hist(human_data, alpha=0.5, label='Human', bins=20, color='blue')
        plt.title(f'{feature}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(base_dir /'feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# plot_feature_analysis(features_df.copy(), df['label'])

#  Data prep
X_text = df["text_clean"]
X_features = features_df.drop('label', axis=1, errors='ignore')
y = df["label"]

# Data split
X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
)

# Improved TF-IDF vectorization 
# Multiple vectorizers to capture different aspects
vectorizer_words = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    stop_words=stop_words,
    min_df=2,  # Ignore words that appear less than 2 times
    max_df=0.95,  # Ignore words that are too frequent
    sublinear_tf=True  # Logarithmic normalization
)

vectorizer_chars = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),  # Character n-grams
    max_features=2000,
    min_df=2
)

# Vectorization
X_text_train_words = vectorizer_words.fit_transform(X_text_train)
X_text_test_words = vectorizer_words.transform(X_text_test)

X_text_train_chars = vectorizer_chars.fit_transform(X_text_train)
X_text_test_chars = vectorizer_chars.transform(X_text_test)

# Normalization of linguistic features
scaler = StandardScaler()
X_feat_train_scaled = scaler.fit_transform(X_feat_train)
X_feat_test_scaled = scaler.transform(X_feat_test)

# Combination of all features
from scipy.sparse import hstack, csr_matrix

X_train_combined = hstack([
    X_text_train_words,
    X_text_train_chars,
    csr_matrix(X_feat_train_scaled)
])

X_test_combined = hstack([
    X_text_test_words,
    X_text_test_chars,
    csr_matrix(X_feat_test_scaled)
])

print(f"Final dimensions: Train {X_train_combined.shape}, Test {X_test_combined.shape}")

# Multiple models with optimization 
print("Training models...")

# Logistic Regression with optimization
lr_params = {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
lr = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=5, scoring='roc_auc')

# Random Forest
rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='roc_auc')

# SVM
svm_params = {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
svm = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=5, scoring='roc_auc')

# Training
models = {'Logistic Regression': lr, 'Random Forest': rf, 'SVM': svm}
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_combined, y_train)
    trained_models[name] = model
    print(f"Best parameters {name}: {model.best_params_}")

# Ensemble voting 
voting_clf = VotingClassifier(
    estimators=[
        ('lr', trained_models['Logistic Regression'].best_estimator_),
        ('rf', trained_models['Random Forest'].best_estimator_),
        ('svm', trained_models['SVM'].best_estimator_)
    ],
    voting='soft'
)

voting_clf.fit(X_train_combined, y_train)

#  Complete evaluation 
def evaluate_model(model, X_test, y_test, model_name):
    """Complete model evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(base_dir /f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

# Evaluation of all models
for name, model in trained_models.items():
    evaluate_model(model, X_test_combined, y_test, name)

evaluate_model(voting_clf, X_test_combined, y_test, "Ensemble Voting")

# Important features analysis :
def analyze_important_features():
    """Analysis of the most important features"""
    # Features from the logistic model
    lr_model = trained_models['Logistic Regression'].best_estimator_
    feature_names = (list(vectorizer_words.get_feature_names_out()) + 
                    list(vectorizer_chars.get_feature_names_out()) + 
                    list(X_feat_train.columns))
    
    coefficients = lr_model.coef_[0]
    feature_importance = list(zip(feature_names, coefficients))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\n=== Top 20 most important features ===")
    for i, (feature, coef) in enumerate(feature_importance[:20]):
        direction = "AI" if coef > 0 else "Human"
        print(f"{i+1:2d}. {feature[:50]:50s} {coef:8.4f} -> {direction}")

analyze_important_features()

#  Cross validation 
print("\n=== Cross Validation ===")
cv_scores = cross_val_score(voting_clf, X_train_combined, y_train, cv=5, scoring='roc_auc')
print(f"CV AUC scores: {cv_scores}")
print(f"CV AUC mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

models_to_save = {
    'vectorizer_words': vectorizer_words,
    'vectorizer_chars': vectorizer_chars,
    'scaler': scaler,
    'voting_classifier': voting_clf,
    'best_lr': trained_models['Logistic Regression'].best_estimator_,
    'best_rf': trained_models['Random Forest'].best_estimator_,
    'best_svm': trained_models['SVM'].best_estimator_
}

for name, model in models_to_save.items():
    with open(base_dir /f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {base_dir / f'{name}.pkl'}")