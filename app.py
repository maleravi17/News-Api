from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import logging
import os
import re
import asyncio
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

def initialize_gemini():
    """Initialize Gemini API with the current API key."""
    global current_key_index
    try:
        if not API_KEYS[current_key_index]:
            logger.error(f"API key at index {current_key_index} is missing")
            raise Exception(f"API key at index {current_key_index} is required")
        logger.info(f"Using API key at index {current_key_index}")
        genai.configure(api_key=API_KEYS[current_key_index])
        logger.info("Gemini API configured successfully")
        model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to configure Gemini API with key at index {current_key_index}: {e}")
        raise

def rotate_key():
    """Rotate to the next API key."""
    global current_key_index
    if current_key_index < len(API_KEYS) - 1:
        current_key_index += 1
        logger.info(f"Rotating to API key at index {current_key_index}")
        return initialize_gemini()
    else:
        logger.error("All API keys have been used")
        raise HTTPException(status_code=500, detail="All API keys have been used. Please add more keys.")

# Initialize Gemini API
try:
    model = initialize_gemini()
except Exception as e:
    logger.error(f"Initial Gemini API configuration failed: {e}")
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

# Function to fetch news links using Gemini API
async def fetch_news_links(query: str) -> List[Dict[str, str]]:
    global model
    attempts = 0
    max_attempts = len(API_KEYS)
    
    while attempts < max_attempts:
        try:
            prompt = f"""
            Perform a web search for up to 15 recent Indian law news articles or legal websites relevant to the query: "{query}".
            Focus on reputable sources like LiveLaw, Bar & Bench, The Hindu (legal section), or IndianKanoon, published within the last 6 months.
            Return the response as a plain text list of articles, with each article formatted as:
            Title: <article title>
            Link: <full URL starting with https://>
            Separate each article with a blank line.
            Example:
            Title: Supreme Court Ruling on Property Law
            Link: https://www.livelaw.in/example
            Title: New Criminal Law Reforms
            Link: https://www.barandbench.com/example
            
            If no articles are found, return an empty response.
            **Return only the formatted text list. Do not include extra text, markdown, code blocks, or JSON.**
            """
            response = model.generate_content(prompt)
            if not response.text:
                logger.error("Gemini API returned empty response")
                return []
            
            # Log the raw response for debugging
            logger.info(f"Raw Gemini response: {response.text}")
            
            # Parse the text response
            articles = []
            article_blocks = response.text.strip().split("\n\n")
            for block in article_blocks:
                lines = block.strip().split("\n")
                title = None
                link = None
                for line in lines:
                    if line.startswith("Title:"):
                        title = line.replace("Title:", "").strip()
                    elif line.startswith("Link:"):
                        link = line.replace("Link:", "").strip()
                if title and link and re.match(r"https?://", link):
                    articles.append({"title": title, "link": link})
                else:
                    logger.warning(f"Invalid article block skipped: {block}")
            
            logger.info(f"Fetched {len(articles)} valid articles from Gemini API")
            return articles
        except Exception as e:
            logger.error(f"Error fetching news from Gemini API: {e}, Query: {query}")
            attempts += 1
            if attempts < max_attempts:
                model = rotate_key()
                await asyncio.sleep(1)  # Add delay to avoid rate-limiting
                continue
            return []
    return []

# Function to rank articles based on query
def rank_articles(query: str, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
    try:
        if not articles:
            logger.warning("No articles to rank")
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
        logger.info(f"Ranked {len(ranked_articles)} articles for query: {query}")
        return [{"title": article["title"], "link": article["link"]} for article in ranked_articles[:10]]
    except Exception as e:
        logger.error(f"Error ranking articles: {e}")
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
            logger.warning("No articles fetched from Gemini API, returning empty list")
            return [{"title": "No articles found", "link": ""}]
        
        # Rank articles
        ranked_articles = rank_articles(query.text, articles)
        
        if not ranked_articles:
            logger.warning("No relevant articles found for query")
            return [{"title": "No articles found", "link": ""}]
        
        return ranked_articles
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in recommend_news: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
