import json
import requests
import feedparser
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime
from collections import Counter

class EconomicScraper:
    """
    Economic Articles Scraper with working sources and error handling
    """
    
    def __init__(self, 
                 database_filename="database_economic_articles.json",
                 min_words=120,  # Lowered from 150
                 max_articles_per_source=30,
                 data_folder="economic_articles"):
        """
        Initialize the improved scraper
        """
        self.database_filename = database_filename
        self.min_words = min_words
        self.max_articles_per_source = max_articles_per_source
        self.data_folder = data_folder
        self.database_path = f"{data_folder}/{database_filename}" if data_folder else database_filename
        
        # Working sources based on diagnostic results
        self.sources = [
            # EXISTING WORKING SOURCES
            {
                'name': 'Economist-Finance',
                'rss': 'https://www.economist.com/finance-and-economics/rss.xml',
                'limit': max_articles_per_source,
                'selectors': ['.article__body-text', '.article-content', 'main'],
                'language': 'English'
            },
            {
                'name': 'VoxEU',
                'rss': 'https://voxeu.org/rss.xml',
                'limit': max_articles_per_source,
                'selectors': ['.field-item', '.content', 'main'],
                'language': 'English'
            },
            {
                'name': 'Project-Syndicate',
                'rss': 'https://www.project-syndicate.org/rss',
                'filter_keywords': ['economic', 'monetary', 'inflation', 'bank', 'finance', 'market', 'trade'],
                'limit': max_articles_per_source,
                'selectors': ['.paywall--article-content', '.article-body', '.content'],
                'language': 'English'
            },
            
            # FIXED CNBC
            {
                'name': 'CNBC-Economy-Fixed',
                'rss': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069',
                'limit': max_articles_per_source,
                'selectors': ['.ArticleBody-articleBody', '.InlineContent-container', '.group', 'div'],
                'language': 'English',
                'special_handler': 'cnbc'
            },
            
            # NEW WORKING SOURCES
            {
                'name': 'BBC-Business',
                'rss': 'http://feeds.bbci.co.uk/news/business/rss.xml',
                'limit': max_articles_per_source,
                'selectors': ['.story-body', '.article-body', 'main'],
                'language': 'English'
            },
            {
                'name': 'Guardian-Economics',
                'rss': 'https://www.theguardian.com/business/economics/rss',
                'limit': max_articles_per_source,
                'selectors': ['.article-body-viewer-selector', '.content__article-body', 'main'],
                'language': 'English'
            },
            {
                'name': 'Bloomberg-Economics',
                'rss': 'https://feeds.bloomberg.com/economics/news.rss',
                'limit': max_articles_per_source,
                'selectors': ['.story-body', '.article-content', 'main'],
                'language': 'English'
            },
            {
                'name': 'NPR-Economy',
                'rss': 'https://feeds.npr.org/1017/rss.xml',
                'limit': max_articles_per_source,
                'selectors': ['.storytext', '.article-body', 'main'],
                'language': 'English'
            }
        ]
        
        # Add French sources
        self._add_french_sources()
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def _add_french_sources(self):
        """Add French economic sources"""
        french_sources = [
            {
                'name': 'LesEchos-Economie',
                'rss': 'https://services.lesechos.fr/rss/les-echos-economie.xml',
                'limit': self.max_articles_per_source,
                'selectors': ['article', 'main article', '.content'],
                'language': 'French'
            },
            {
                'name': 'LaTribune-Economie',
                'rss': 'https://www.latribune.fr/economie/rss',
                'limit': self.max_articles_per_source,
                'selectors': ['.article-content', '.content', 'main'],
                'language': 'French'
            },
            {
                'name': 'LeFigaro-Economie',
                'rss': 'https://www.lefigaro.fr/rss/figaro_economie.xml',
                'limit': self.max_articles_per_source,
                'selectors': ['.fig-content-body', '.article-content', 'main'],
                'language': 'French'
            },
            {
                'name': 'Liberation-Economie',
                'rss': 'https://www.liberation.fr/arc/outboundfeeds/rss/category/economie/',
                'limit': self.max_articles_per_source,
                'selectors': ['.article-body', '.content', 'main'],
                'language': 'French'
            },
            {
                'name': 'Challenges-Economie',
                'rss': 'https://www.challenges.fr/rss/economie.xml',
                'limit': self.max_articles_per_source,
                'selectors': ['.article-content', '.content', 'main'],
                'language': 'French'
            },
            {
                'name': 'FranceInfo-Eco',
                'rss': 'https://francetvinfo.fr/economie.rss',
                'limit': self.max_articles_per_source,
                'selectors': ['.article-content', '.content', 'main'],
                'language': 'French'
            }
        ]
        
        self.sources.extend(french_sources)
    
    def test_french_sources(self):
        """Test French sources availability"""
        print("TESTING FRENCH ECONOMIC SOURCES")
        print("=" * 40)
        
        french_sources = [s for s in self.sources if s.get('language') == 'French']
        working_french = []
        
        for source in french_sources:
            try:
                print(f"\nTesting {source['name']}:")
                response = requests.get(source['rss'], timeout=10)
                print(f"  HTTP Status: {response.status_code}")
                
                if response.status_code == 200:
                    feed = feedparser.parse(response.text)
                    entries = len(feed.entries)
                    print(f"  Entries: {entries}")
                    
                    if entries > 0:
                        print(f"  Working - {feed.feed.get('title', 'No title')}")
                        working_french.append(source['name'])
                    else:
                        print(f"  No entries")
                else:
                    print(f"  Failed")
            except Exception as e:
                print(f"  Error: {str(e)[:50]}...")
        
        print(f"\nFRENCH SOURCES SUMMARY:")
        print(f"  Working: {len(working_french)}/{len(french_sources)}")
        for source_name in working_french:
            print(f"    Working: {source_name}")
        
        return working_french
    
    def scrape_cnbc_article(self, url):
        """Special handler for CNBC articles"""
        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements more aggressively
            for unwanted in soup.select('script, style, nav, footer, aside, .ad, .advertisement, .social-share'):
                unwanted.decompose()
            
            # CNBC-specific approach
            content_text = ""
            
            # Try CNBC-specific selectors first
            cnbc_selectors = [
                '.ArticleBody-articleBody',
                '.InlineContent-container', 
                '.RenderKeyPoints-list',
                '[data-module="ArticleBody"]'
            ]
            
            for selector in cnbc_selectors:
                elements = soup.select(selector)
                if elements:
                    text = elements[0].get_text(separator=' ', strip=True)
                    if len(text) > len(content_text):
                        content_text = text
            
            # If still not enough content, try a more aggressive approach
            if len(content_text.split()) < 100:
                # Get all divs with substantial content
                all_divs = soup.find_all('div')
                substantial_texts = []
                
                for div in all_divs:
                    div_text = div.get_text(strip=True)
                    # Skip divs that are too short or look like navigation
                    if (len(div_text) > 50 and 
                        not any(nav_word in div_text.lower() for nav_word in ['menu', 'navigation', 'footer', 'header', 'cookie'])):
                        substantial_texts.append(div_text)
                
                if substantial_texts:
                    # Take the longest substantial text blocks
                    substantial_texts.sort(key=len, reverse=True)
                    content_text = ' '.join(substantial_texts[:3])  # Top 3 longest blocks
            
            # Final fallback to paragraphs
            if len(content_text.split()) < 50:
                paragraphs = soup.find_all('p')
                para_texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
                if para_texts:
                    content_text = ' '.join(para_texts)
            
            # Clean and return
            content_text = re.sub(r'\\s+', ' ', content_text).strip()
            return content_text
            
        except Exception as e:
            print(f"      CNBC scraping error: {str(e)[:50]}...")
            return None
    
    def scrape_article_content(self, url, source_config):
        """Improved article content scraping with source-specific handling"""
        
        # Use special handler for CNBC
        if source_config.get('special_handler') == 'cnbc':
            return self.scrape_cnbc_article(url)
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Clean unwanted elements
            for unwanted in soup.select('nav, footer, aside, .ad, .advertisement, script, style, .social, .share, .cookie'):
                unwanted.decompose()
            
            # Try source-specific selectors first
            content_text = ""
            best_length = 0
            
            selectors = source_config.get('selectors', [
                '.article-body', '.article-content', '.story-body', '.post-content',
                '.entry-content', '.content', 'main', '.field-item', '.text-content'
            ])
            
            for selector in selectors:
                elem = soup.select_one(selector)
                if elem:
                    text = elem.get_text(separator=' ', strip=True)
                    if len(text) > best_length:
                        content_text = text
                        best_length = len(text)
            
            # Fallback to paragraphs if content is too short
            if best_length < 400:
                paragraphs = soup.find_all('p')
                para_texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]
                if para_texts:
                    fallback_content = ' '.join(para_texts)
                    if len(fallback_content) > best_length:
                        content_text = fallback_content
            
            # Clean text
            content_text = re.sub(r'\\s+', ' ', content_text).strip()
            return content_text
            
        except Exception as e:
            print(f"      ERROR scraping {url}: {str(e)[:50]}...")
            return None
    
    def scrape_from_source(self, source):
        """Scrape articles from a single source with improved error handling"""
        print(f"\\n{source['name']} ({source.get('language', 'Unknown')})...")
        articles = []
        
        try:
            feed = feedparser.parse(source['rss'])
            print(f"   {len(feed.entries)} entries found")
            
            if len(feed.entries) == 0:
                print(f"   WARNING: No entries found")
                return articles
            
            success_count = 0
            attempts = 0
            
            for entry in feed.entries:
                if success_count >= source['limit']:
                    break
                
                attempts += 1
                if attempts > source['limit'] * 2:  # Safety limit
                    break
                    
                title = entry.get('title', '')
                link = entry.get('link', '')
                summary = entry.get('summary', '')
                
                if not title or not link:
                    continue
                
                # Apply keyword filtering if specified
                if source.get('filter_keywords'):
                    text_to_check = f"{title} {summary}".lower()
                    if not any(keyword in text_to_check for keyword in source['filter_keywords']):
                        continue
                
                print(f"   [{success_count+1}] {title[:50]}...")
                
                # Scrape article content
                content_text = self.scrape_article_content(link, source)
                
                if content_text:
                    word_count = len(content_text.split())
                    
                    if word_count >= self.min_words:
                        articles.append({
                            'title': title.strip(),
                            'url': link,
                            'source': source['name'],
                            'content': content_text,
                            'word_count': word_count,
                            'length': len(content_text),
                            'content_type': 'Economic Article',
                            'language': source.get('language', 'Unknown'),
                            'date': entry.get('published', ''),
                            'summary': summary[:200] if summary else '',
                            'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        print(f"      SUCCESS ({word_count} words)")
                        success_count += 1
                    else:
                        print(f"      SKIP: Too short ({word_count} words, need {self.min_words})")
                else:
                    print(f"      SKIP: No content extracted")
                
                time.sleep(1.0)  # Slightly faster but still respectful
                
        except Exception as e:
            print(f"   ERROR with source {source['name']}: {str(e)[:50]}...")
        
        print(f"   Collected {len(articles)} articles from {source['name']}")
        return articles
    
    def load_database(self):
        """Load existing database"""
        try:
            with open(self.database_path, "r", encoding="utf-8") as f:
                database = json.load(f)
                articles = database.get('articles', [])
                print(f"Loaded existing database: {len(articles)} articles")
                return articles
        except FileNotFoundError:
            print(f"No existing database found at {self.database_path}, starting fresh")
            return []
    
    def save_database(self, articles):
        """Save database with enhanced metadata"""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(url)
        
        # Enhanced metadata
        language_counts = Counter(article.get('language', 'Unknown') for article in unique_articles)
        source_counts = Counter(article.get('source', 'Unknown') for article in unique_articles)
        
        database_info = {
            "metadata": {
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_articles": len(unique_articles),
                "total_words": sum(article.get('word_count', 0) for article in unique_articles),
                "min_words_threshold": self.min_words,
                "max_articles_per_source": self.max_articles_per_source,
                "languages": dict(language_counts),
                "sources": dict(source_counts),
                "scraper_version": "improved_v2"
            },
            "articles": unique_articles
        }
        
        with open(self.database_path, "w", encoding="utf-8") as f:
            json.dump(database_info, f, ensure_ascii=False, indent=2)
        
        print(f"\\nDatabase saved to {self.database_path}: {len(unique_articles)} articles")
        return database_info
    
    def update_database(self):
        """Main method with improved progress tracking"""
        print("IMPROVED ECONOMIC ARTICLES SCRAPER")
        print("=" * 60)
        print(f"Database: {self.database_path}")
        print(f"Min words: {self.min_words}")
        print(f"Max per source: {self.max_articles_per_source}")
        print(f"Total sources configured: {len(self.sources)}")
        
        # Show language distribution of sources
        lang_counts = Counter(s.get('language', 'Unknown') for s in self.sources)
        print(f"Languages: {dict(lang_counts)}")
        
        # Load existing articles
        existing_articles = self.load_database()
        print(f"\\nStarting with {len(existing_articles)} existing articles")
        
        # Scrape new articles
        print(f"\\nScraping from {len(self.sources)} sources...")
        new_articles = []
        
        for i, source in enumerate(self.sources, 1):
            print(f"\\n[{i}/{len(self.sources)}] Processing {source['name']}...")
            source_articles = self.scrape_from_source(source)
            new_articles.extend(source_articles)
        
        print(f"\\nSCRAPING COMPLETE:")
        print(f"   New articles scraped: {len(new_articles)}")
        print(f"   Total before deduplication: {len(existing_articles) + len(new_articles)}")
        
        # Combine and save
        all_articles = existing_articles + new_articles
        final_database = self.save_database(all_articles)
        
        # Enhanced final statistics
        final_count = final_database['metadata']['total_articles']
        final_words = final_database['metadata']['total_words']
        languages = final_database['metadata']['languages']
        
        print(f"\\nFINAL STATISTICS:")
        print(f"   Total articles: {final_count}")
        print(f"   Total words: {final_words:,}")
        print(f"   Average words/article: {final_words // final_count if final_count else 0}")
        print(f"   Languages: {languages}")
        
        print(f"\\nTop sources:")
        for source, count in sorted(final_database['metadata']['sources'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {source}: {count} articles")
        
        return final_database
    
    def list_sources_by_language(self):
        """List sources organized by language"""
        print("CONFIGURED SOURCES BY LANGUAGE")
        print("=" * 40)
        
        lang_groups = {}
        for source in self.sources:
            lang = source.get('language', 'Unknown')
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append(source)
        
        for lang, sources in lang_groups.items():
            print(f"\\n{lang} ({len(sources)} sources):")
            for source in sources:
                keywords = source.get('filter_keywords', [])
                keywords_str = f" (keywords: {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''})" if keywords else ""
                special = f" [SPECIAL: {source['special_handler']}]" if source.get('special_handler') else ""
                print(f"   {source['name']} - max {source['limit']}{keywords_str}{special}")

# Convenience function
def create_improved_scraper(min_words=120, max_per_source=30):
    """Create improved scraper with tested sources"""
    return EconomicScraper(
        min_words=min_words,
        max_articles_per_source=max_per_source
    )

if __name__ == "__main__":
    # Create improved scraper
    scraper = create_improved_scraper(min_words=120, max_per_source=25)
    
    # Test French sources first
    print("Testing French sources availability...")
    working_french = scraper.test_french_sources()
    
    # Show all sources
    scraper.list_sources_by_language()
    
    # Ask user if they want to proceed
    proceed = input(f"\\nProceed with scraping? ({len(working_french)} French sources working) [y/N]: ")
    
    if proceed.lower() == 'y':
        # Update database
        database = scraper.update_database()
        
        final_count = database['metadata']['total_articles']
        if final_count >= 250:
            print(f"\\nEXCELLENT! {final_count} articles - Ready for ML training!")
        else:
            print(f"\\n{final_count} articles collected. Consider adjusting parameters for more.")
    else:
        print("Scraping cancelled.")