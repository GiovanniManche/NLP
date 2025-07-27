# NLP

The aim of this projet is to train an NLP classifier capable of distinguishing whether a given economic text (article, report, speech, etc.) originates from a real economic figure (such as the ECB, the FED, economist columns, or academic papers) or from an AI-generated source.

1) Data scraping

   
To generate real data, we used a web scraper. It collects economic and financial news articles from multiple English and French sources using their RSS feeds (e.g., BBC, Bloomberg, Les Échos, La Tribune). For each article, the scraper follows the link, extracts the main text content with the help of BeautifulSoup, and applies filters to remove advertisements, navigation menus, and irrelevant elements. It also enforces a minimum word count (e.g., 120 words) and avoids duplicate articles.

To generate synthetic AI data, we used an asynchronous text generation pipeline. It reads economic topics from a real dataset (data that we scrapped) and rapidly generates AI-written texts using the Groq API and the LLaMA 3.1 8B instant model. The generator builds prompts in French or English, optionally formatted as economic speeches or articles, and sends them concurrently (up to 25 simultaneous requests) to maximize speed.
