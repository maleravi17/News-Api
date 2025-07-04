from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import logging
import os
import re
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# API key rotation
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]
current_key_index = 0
current_model_name = "gemini-2.0-flash-001"

def initialize_gemini():
    """Initialize Gemini API with the current API key and model."""
    global current_key_index, current_model_name
    try:
        if not API_KEYS[current_key_index]:
            logger.error("Missing API key at index %s", current_key_index)
            raise Exception("API key missing")
        logger.info("Using API key at index %s for model %s", current_key_index, current_model_name)
        genai.configure(api_key=API_KEYS[current_key_index])
        model = genai.GenerativeModel(current_model_name)
        logger.info("Gemini model %s initialized successfully", current_model_name)
        return model
    except Exception as e:
        logger.error("Failed to initialize Gemini API with key at index %s for model %s: %s", current_key_index, current_model_name, str(e))
        raise

def rotate_key():
    """Rotate to the next API key."""
    global current_key_index
    if current_key_index < len(API_KEYS) - 1:
        current_key_index += 1
        logger.info("Rotating to API key at index %s", current_key_index)
        return initialize_gemini()
    else:
        logger.error("All API keys have been used")
        raise HTTPException(status_code=500, detail="All API keys exhausted")

# Initialize Gemini API
try:
    model = initialize_gemini()
except Exception as e:
    logger.error("Initial Gemini API configuration failed: %s", str(e))
    raise Exception("Gemini API configuration failed")

# Root endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to the Indian Law News Recommendation API"}

@app.head("/")
async def root_head():
    return JSONResponse(content={}, status_code=200)

# Pydantic model for input payload
class UserQuery(BaseModel):
    text: str

async def fetch_news_links(query: str) -> List[Dict[str, str]]:
    global model, current_model_name
    attempts = 0
    max_attempts = max(len(API_KEYS), 1)
    models_to_try = ["gemini-2.0-flash-001", "gemini-1.5-pro"]
    
    for model_name in models_to_try:
        current_model_name = model_name
        logger.info("Attempting to fetch news with model %s", current_model_name)
        model = initialize_gemini()
        attempts = 0
        
        while attempts < max_attempts:
            try:
                prompt = """
                Perform a web search for up to 15 recent Indian law news articles relevant to the query: "{}".
                Focus on reputable sources like LiveLaw, Bar & Bench, The Hindu, or IndianKanoon, published within the last 6 months.
                Return a JSON list of objects, each containing:
                - "title": The article title
                - "link": The full URL, starting with https://
                Ensure links are valid and point to specific articles.
                Format as a JSON list: [{"title": "example", "link": "https://example.com"}, ...]
                If no articles are found, return: []
                """.format(query)
                logger.info("Sending prompt to Gemini API for query %s", query)
                response = model.generate_content(prompt)
                
                if not response.text:
                    logger.error("Gemini API returned empty response for query %s with model %s", query, current_model_name)
                    return []
                
                # Log raw response
                logger.info("Gemini API raw response for query %s with model %s: %s", query, current_model_name, response.text)
                
                # Parse JSON response
                try:
                    cleaned_response = response.text.strip("```json\n```").strip()
                    articles = json.loads(cleaned_response)
                    if not isinstance(articles, list):
                        logger.error("Gemini API response is not a JSON list for query %s with model %s: %s", query, current_model_name, cleaned_response)
                        return []
                    
                    # Validate articles
                    valid_articles = []
                    for article in articles:
                        if isinstance(article, dict) and "title" in article and "link" in article:
                            title = article["title"].strip()
                            link = article["link"].strip()
                            if re.match(r"https?://", link):
                                valid_articles.append({"title": title, "link": link})
                            else:
                                logger.warning("Invalid URL skipped for query %s: %s", query, link)
                        else:
                            logger.warning("Invalid article format for query %s: %s", query, str(article))
                    
                    logger.info("Fetched %s valid articles for query %s with model %s", len(valid_articles), query, current_model_name)
                    return valid_articles
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse Gemini API response for query %s with model %s: %s, Raw response: %s", query, current_model_name, str(e), response.text)
                    return []
            except Exception as e:
                logger.error("Error fetching news from Gemini API for query %s with model %s: %s", query, current_model_name, str(e))
                attempts += 1
                if attempts < max_attempts:
                    model = rotate_key()
                    continue
                break
        if model_name == models_to_try[-1]:
            logger.error("All models and API keys failed for query %s", query)
            return []
    
    return []

def rank_articles(query: str, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
    try:
        if not articles:
            logger.warning("No articles to rank for query %s", query)
            return []
        
        titles = [article["title"] for article in articles]
        links = [article["link"] for article in articles]
        corpus = titles + [query]
        
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = tfidf_matrix[-1]
        article_vectors = tfidf_matrix[:-1]
        similarities = (query_vector * article_vectors.T).toarray()[0]
        
        ranked_articles = []
        for idx, sim in enumerate(similarities):
            ranked_articles.append({"title": titles[idx], "link": links[idx], "score": float(sim)})
        
        ranked_articles.sort(key=lambda x: x["score"], reverse=True)
        logger.info("Ranked %s articles for query %s", len(ranked_articles), query)
        return [{"title": article["title"], "link": article["link"]} for article in ranked_articles[:10]]
    except Exception as e:
        logger.error("Error ranking articles for query %s: %s", query, str(e))
        return []

@app.post("/recommend", response_model=List[Dict[str, str]])
async def recommend_news(query: UserQuery):
    try:
        if not query.text.strip():
            logger.error("Empty query provided")
            raise HTTPException(status_code=400, detail="Query text cannot be empty")

        # Fetch articles using Gemini API
        articles = await fetch_news_links(query.text)
        
        if not articles:
            logger.warning("No articles fetched from Gemini API for query %s", query.text)
            return []
        
        # Rank articles
        ranked_articles = rank_articles(query.text, articles)
        
        if not ranked_articles:
            logger.warning("No relevant articles found for query %s", query.text)
            return []
        
        return ranked_articles
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Unexpected error in recommend_news for query %s: %s", query.text, str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
