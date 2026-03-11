"""
Configuration module for Smart-Travel RAG Itinerary Generator.
Manages API keys, model settings, and application constants.
"""

import os

# ─────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────
# Set your Gemini API key as an environment variable:
#   Windows:  set GEMINI_API_KEY=your_api_key_here
#   Linux:    export GEMINI_API_KEY=your_api_key_here
# Or create a .env file with: GEMINI_API_KEY=your_api_key_here

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ─────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────
MODEL_NAME = "gemini-2.0-flash"
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

# ─────────────────────────────────────────────
# Retrieval Configuration
# ─────────────────────────────────────────────
TOP_K_RESULTS = 8          # Number of top matching sentences to retrieve
MIN_RELEVANCE_SCORE = 0.01 # Minimum TF-IDF score to consider a sentence relevant

# ─────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────
KNOWLEDGE_BASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "travel_data.txt")

# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert local travel guide for Jammu & Kashmir, India. 
You have deep knowledge of the region's destinations, culture, cuisine, history, and travel logistics.
Your responses should be warm, helpful, and detailed.
Always structure itineraries clearly with day-by-day plans when applicable.
Include practical tips like best time to visit, how to reach, what to carry, and local food recommendations.
If the context doesn't contain enough information for a specific query, use your general knowledge but mention 
that the recommendation is based on general knowledge rather than verified local information."""

QUERY_PROMPT_TEMPLATE = """Based on the following verified travel information about Jammu & Kashmir:

--- TRAVEL KNOWLEDGE BASE ---
{context}
--- END OF KNOWLEDGE BASE ---

User's travel query: {query}

Please provide a detailed, well-structured response to the user's query. 
If generating an itinerary, organize it day-by-day with specific activities, timings, and travel tips.
Include relevant details about food, accommodation suggestions, and practical travel advice."""

# ─────────────────────────────────────────────
# Stop Words (common English words to ignore during retrieval)
# ─────────────────────────────────────────────
STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "not", "with", "by", "from", "as", "this", "that", "these",
    "those", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "am", "are", "was", "were", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "his", "her", "its", "what", "which", "who",
    "how", "when", "where", "why", "all", "each", "some", "any", "no", "if",
    "about", "up", "out", "so", "than", "too", "very", "just", "also",
    "into", "over", "after", "before", "between", "through", "during",
    "plan", "suggest", "tell", "give", "show", "want", "need", "like",
    "please", "help", "know", "find", "make", "go", "see", "get",
}
