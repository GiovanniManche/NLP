# data_diagnostic.py - Diagnostic tool to find why 100% accuracy occurs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from wordcloud import WordCloud

class DataLeakageDetector:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, sep=";", encoding="ISO-8859-1")
        print(f"Dataset loaded: {len(self.df)} texts")
        print(f"Distribution: {self.df['label'].value_counts().to_dict()}")
    
    def basic_statistics(self):
        """Basic statistics to detect obvious patterns"""
        print("\nBASIC STATISTICS")
        print("="*50)
        
        # Statistics by class
        for label in [0, 1]:
            subset = self.df[self.df['label'] == label]
            class_name = "Human" if label == 0 else "AI"
            
            print(f"\nClass {class_name} ({len(subset)} texts):")
            print(f"   Average length: {subset['text'].str.len().mean():.1f} characters")
            print(f"   Median length: {subset['text'].str.len().median():.1f} characters")
            print(f"   Min/Max length: {subset['text'].str.len().min()}-{subset['text'].str.len().max()}")
            print(f"   Average words: {subset['text'].str.split().str.len().mean():.1f}")
            
        # Statistical test for length difference
        human_lengths = self.df[self.df['label'] == 0]['text'].str.len()
        ai_lengths = self.df[self.df['label'] == 1]['text'].str.len()
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(human_lengths, ai_lengths)
        print(f"\nLength difference test (t-test):")
        print(f"   p-value: {p_value:.6f}")
        if p_value < 0.05:
            print("   WARNING: SIGNIFICANT length difference between classes!")
        else:
            print("   OK: No significant length difference")
    
    def detect_suspicious_patterns(self):
        """Detects suspicious words/patterns that reveal the source"""
        print("\nSEARCHING FOR SUSPICIOUS PATTERNS")
        print("="*50)
        
        # Suspicious words in texts
        suspicious_words = [
            # AI words
            'gpt', 'chatgpt', 'ai', 'artificial intelligence', 'generated', 'assistant', 
            'language model', 'openai', 'claude', 'anthropic', 'model', 'algorithm',
            'neural network', 'deep learning', 'machine learning', 'automated',
            
            # AI style patterns
            'in conclusion', 'furthermore', 'moreover', 'it is important to note',
            'however', 'therefore', 'additionally', 'on the other hand',
            
            # Human sources
            'author', 'columnist', 'journalist', 'editor', 'reporter', 'source',
            'interview', 'exclusive', 'breaking news', 'correspondent'
        ]
        
        suspicious_found = []
        
        for word in suspicious_words:
            human_count = self.df[self.df['label'] == 0]['text'].str.lower().str.contains(word, na=False).sum()
            ai_count = self.df[self.df['label'] == 1]['text'].str.lower().str.contains(word, na=False).sum()
            
            if human_count > 0 or ai_count > 0:
                human_pct = human_count / len(self.df[self.df['label'] == 0]) * 100
                ai_pct = ai_count / len(self.df[self.df['label'] == 1]) * 100
                
                # If difference > 10%, it's suspicious
                if abs(human_pct - ai_pct) > 10:
                    suspicious_found.append({
                        'word': word,
                        'human_pct': human_pct,
                        'ai_pct': ai_pct,
                        'difference': abs(human_pct - ai_pct)
                    })
        
        if suspicious_found:
            print("WARNING: SUSPICIOUS WORDS DETECTED:")
            suspicious_found.sort(key=lambda x: x['difference'], reverse=True)
            for item in suspicious_found[:10]:
                print(f"   '{item['word']}': Human {item['human_pct']:.1f}% vs AI {item['ai_pct']:.1f}% "
                      f"(diff: {item['difference']:.1f}%)")
        else:
            print("OK: No obvious suspicious words detected")
    
    def analyze_vocabulary_differences(self):
        """Analyzes vocabulary differences between classes"""
        print("\nVOCABULARY ANALYSIS")
        print("="*50)
        
        # Create separate corpora
        human_texts = ' '.join(self.df[self.df['label'] == 0]['text'].tolist())
        ai_texts = ' '.join(self.df[self.df['label'] == 1]['text'].tolist())
        
        # Most frequent words by class
        def get_top_words(text, n=20):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return Counter(words).most_common(n)
        
        human_top = get_top_words(human_texts)
        ai_top = get_top_words(ai_texts)
        
        print("Top HUMAN words:")
        for word, count in human_top[:10]:
            print(f"   {word}: {count}")
        
        print("\nTop AI words:")
        for word, count in ai_top[:10]:
            print(f"   {word}: {count}")
        
        # Words unique to one class
        human_words = set([word for word, _ in human_top])
        ai_words = set([word for word, _ in ai_top])
        
        only_human = human_words - ai_words
        only_ai = ai_words - human_words
        
        if only_human:
            print(f"\nWords only in HUMAN texts: {list(only_human)[:10]}")
        if only_ai:
            print(f"Words only in AI texts: {list(only_ai)[:10]}")
    
    def visualize_text_distributions(self):
        """Creates visualizations to detect patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Length distribution
        human_lengths = self.df[self.df['label'] == 0]['text'].str.len()
        ai_lengths = self.df[self.df['label'] == 1]['text'].str.len()
        
        axes[0,0].hist(human_lengths, alpha=0.7, label='Human', bins=30, color='blue')
        axes[0,0].hist(ai_lengths, alpha=0.7, label='AI', bins=30, color='red')
        axes[0,0].set_title('Text Length Distribution')
        axes[0,0].set_xlabel('Length (characters)')
        axes[0,0].legend()
        
        # Word count distribution
        human_words = self.df[self.df['label'] == 0]['text'].str.split().str.len()
        ai_words = self.df[self.df['label'] == 1]['text'].str.split().str.len()
        
        axes[0,1].hist(human_words, alpha=0.7, label='Human', bins=30, color='blue')
        axes[0,1].hist(ai_words, alpha=0.7, label='AI', bins=30, color='red')
        axes[0,1].set_title('Word Count Distribution')
        axes[0,1].set_xlabel('Number of words')
        axes[0,1].legend()
        
        # Average sentence length
        def avg_sentence_length(text):
            sentences = re.split(r'[.!?]+', text)
            if len(sentences) > 1:
                return np.mean([len(s.split()) for s in sentences if s.strip()])
            return len(text.split())
        
        human_sent_len = self.df[self.df['label'] == 0]['text'].apply(avg_sentence_length)
        ai_sent_len = self.df[self.df['label'] == 1]['text'].apply(avg_sentence_length)
        
        axes[1,0].hist(human_sent_len, alpha=0.7, label='Human', bins=20, color='blue')
        axes[1,0].hist(ai_sent_len, alpha=0.7, label='AI', bins=20, color='red')
        axes[1,0].set_title('Average Sentence Length')
        axes[1,0].set_xlabel('Words per sentence')
        axes[1,0].legend()
        
        # PCA on TF-IDF to see separation
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(self.df['text'])
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())
        
        human_pca = X_pca[self.df['label'] == 0]
        ai_pca = X_pca[self.df['label'] == 1]
        
        axes[1,1].scatter(human_pca[:, 0], human_pca[:, 1], alpha=0.6, label='Human', color='blue')
        axes[1,1].scatter(ai_pca[:, 0], ai_pca[:, 1], alpha=0.6, label='AI', color='red')
        axes[1,1].set_title('PCA of TF-IDF Features')
        axes[1,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('BERT_bias_study/plots/data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # PCA interpretation
        if len(set(X_pca[self.df['label'] == 0, 0])) == 1 or len(set(X_pca[self.df['label'] == 1, 0])) == 1:
            print("WARNING: PERFECT SEPARATION in PCA space - Data leakage confirmed!")
        else:
            print("OK: No obvious perfect separation in PCA space")
    
    def show_examples(self, n=3):
        """Shows examples from each class for manual inspection"""
        print(f"\nTEXT EXAMPLES (manual inspection)")
        print("="*50)
        
        print("HUMAN TEXTS:")
        human_samples = self.df[self.df['label'] == 0].sample(n, random_state=42)
        for i, (_, row) in enumerate(human_samples.iterrows()):
            print(f"\nHuman {i+1} ({len(row['text'])} chars):")
            print(f"'{row['text'][:300]}{'...' if len(row['text']) > 300 else ''}'")
        
        print(f"\nAI TEXTS:")
        ai_samples = self.df[self.df['label'] == 1].sample(n, random_state=42)
        for i, (_, row) in enumerate(ai_samples.iterrows()):
            print(f"\nAI {i+1} ({len(row['text'])} chars):")
            print(f"'{row['text'][:300]}{'...' if len(row['text']) > 300 else ''}'")
    
    def run_full_diagnosis(self):
        """Runs all diagnostic tests"""
        print("COMPLETE DATASET DIAGNOSIS")
        print("="*60)
        
        self.basic_statistics()
        self.detect_suspicious_patterns()
        self.analyze_vocabulary_differences()
        self.show_examples()
        
        print(f"\nVisualizations saved in 'data_analysis.png'")
        self.visualize_text_distributions()
        
        print(f"\nCONCLUSION:")
        print("If your models reach 100% accuracy, it's probably due to:")
        print("1. Data leakage (revealing words in texts)")
        print("2. Systematic length/style differences")
        print("3. Dataset too simple/artificial")
        print("4. Repetitive patterns in one class")

if __name__ == "__main__":
    detector = DataLeakageDetector("economic_articles/clean_economic_dataset.csv")
    detector.run_full_diagnosis()