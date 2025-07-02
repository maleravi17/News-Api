from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import logging
import os
import re
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure Gemini API
try:
    ##GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "YOUR_GEMINI_API_KEY"  # Replace with your key or use env variable
    GEMINI_API_KEY="AIzaSyB1liOmkD_MUtLaG502T24D4_mFi1hqItw"
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-pro")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise Exception("Gemini API configuration failed")

# Pydantic model for input payload
class UserQuery(BaseModel):
    text: str

# Function to fetch news links using Gemini API
async def fetch_news_links(query: str) -> List[Dict[str, str]]:
    try:
        prompt = f"""
        Provide a list of up to 15 recent Indian law news articles or legal websites relevant to the query: "{query}".
        Summarize this query to a topic and focus on topic-related websites, reputable sources like LiveLaw, Bar & Bench, The Hindu (legal section), or IndianKanoon etc.
        Return the response as a JSON list of objects, each containing:
        - "title": The article or website title
        - "link": The full URL,starting with https://
        Ensure links are valid and point to specific articles or relevant legal news pages.
        Do not include non-legal or outdated sources.
        """
        response = model.generate_content(prompt)
        if not response.text:
            logger.error("Gemini API returned empty response")
            return []
        
        # Parse JSON response
        try:
            articles = json.loads(response.text.strip("```json\n```"))
            if not isinstance(articles, list):
                logger.error("Gemini API response is not a valid JSON list")
                return []
            
            # Validate and clean articles
            valid_articles = []
            for article in articles:
                if isinstance(article, dict) and "title" in article and "link" in article:
                    title = article["title"].strip()
                    link = article["link"].strip()
                    if re.match(r"https?://", link):
                        valid_articles.append({"title": title, "link": link})
                    else:
                        logger.warning(f"Invalid URL skipped: {link}")
                else:
                    logger.warning(f"Invalid article format: {article}")
            
            logger.info(f"Fetched {len(valid_articles)} valid articles from Gemini API")
            return valid_articles
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini API response as JSON: {e}")
            return []
    except Exception as e:
        logger.error(f"Error fetching news from Gemini API: {e}")
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
            logger.error("No articles fetched from Gemini API")
            raise HTTPException(status_code=500, detail="Failed to fetch news articles")
        
        # Rank articles
        ranked_articles = rank_articles(query.text, articles)
        
        if not ranked_articles:
            logger.warning("No relevant articles found for query")
            raise HTTPException(status_code=404, detail="No relevant articles found")
        
        return ranked_articles
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in recommend_news: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
