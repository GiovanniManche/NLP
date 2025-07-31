# NLP Final Project : classifier to differenciate between human-written economic texts and AI-generated ones
![Python](https://img.shields.io/badge/Python-3.12.4-blue)

## Introduction 
**Authors : Tania ADMANE and Giovanni MANCHE** \
**Date** : July 2025 \
**Subject** : Final Project of the course "Introduction to NLP" at University Paris-Dauphine by Mr. Yair Lakretz


## Description 
The aim of this projet is to train an NLP classifier capable of distinguishing whether a given economic text (article, report, speech, etc.) originates from a real economic figure (such as the ECB, the FED, economist columns, or academic papers) or from an AI-generated source. To do so, we : 
- scrap data from several English and French sources (ECB press releases, columns, news), filtering to only get economic and financial subjects and clean it when necessary
- generate the same amount of texts via an AI-model,
- develop baseline classifier models to compare with the primary model,
- implement BERT-family models with classifier layer as our primary models  

## Project architecture and running
### Structure
```
NLP/
├── bias_study/
│   ├── plots/
│   ├── data_diagnostic.py        
├── BERT_model/  # contains the standard BERT used
│   ├── saved_models/
│   │   └── bert_classifier.pth    # To avoid training the model several times 
│   ├── config.py        
│   └── main.py
    └── model.py         
├── DistilBERT_models/  # contains the DistilBERT with and without troncature
│   ├── saved_models/
│   │   └── bert_classifier.pth    # To avoid training the model several times 
│   ├── config.py        
│   └── main.py
|   └── model.py
├── economic_articles/  # contains the webscraper and AI texts generator
│   ├── economic_articles_scraper.py  # webscraping of economic texts        
│   └── IA_generator.py
    └── process_dataset.py        
```

### How to run 
#### Data scraping
One can scrap data from the web simply by running the file `economic_articles_scraper.py`. Adding url is possible and we automatically combine the new texts with the old database.

#### AI generator

#### SVM

#### BERT and DistilBERT 
Both are organized the same way : 
- in `config.py`, one can enter the different paths to its CSV files containing database, as well as model hyperparameters and the type of BERT one wants to use (uncased or not, base or distilled,...)
- in `model.py`, we define how does the model "work". The model's architecture is defined, as well as how to deal with data. 
- `main.py` is the file one should run to get results. It deals with the training and evaluation processes, returns evaluation metrics (accuracy, confusion matrix, F1-score,...). 

Please note that the file `DistilBERT_truncated.py` is independant and can be run separately.
![Console Output](docs/other/BERT_code_structure.png)


## Maybe somethings about the code itself ? Like what it does etc but we already have that in the repoert
1) Data scraping

   
To generate real data, we used a web scraper. It collects economic and financial news articles from multiple English and French sources using their RSS feeds (e.g., BBC, Bloomberg, Les Échos, La Tribune). For each article, the scraper follows the link, extracts the main text content with the help of BeautifulSoup, and applies filters to remove advertisements, navigation menus, and irrelevant elements. It also enforces a minimum word count (e.g., 120 words) and avoids duplicate articles.

To generate synthetic AI data, we used an asynchronous text generation pipeline. It reads economic topics from a real dataset (data that we scrapped) and rapidly generates AI-written texts using the Groq API and the LLaMA 3.1 8B instant model. The generator builds prompts in French or English, optionally formatted as economic speeches or articles, and sends them concurrently (up to 25 simultaneous requests) to maximize speed.
