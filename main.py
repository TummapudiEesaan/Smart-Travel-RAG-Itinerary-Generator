"""
Smart-Travel RAG Itinerary Generator
=====================================
An AI-powered travel assistant that generates personalized travel itineraries
using Retrieval-Augmented Generation (RAG) with Google Gemini API.

Usage:
    python main.py
"""

import os
import re
import sys
import math
import string
from collections import Counter

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional for CLI mode

from config import (
    GEMINI_API_KEY,
    MODEL_NAME,
    GENERATION_CONFIG,
    TOP_K_RESULTS,
    MIN_RELEVANCE_SCORE,
    KNOWLEDGE_BASE_FILE,
    SYSTEM_PROMPT,
    QUERY_PROMPT_TEMPLATE,
    STOP_WORDS,
)


# ═══════════════════════════════════════════════════════════════
# Knowledge Base Module
# ═══════════════════════════════════════════════════════════════

class KnowledgeBase:
    """Loads and manages the travel knowledge base from travel_data.txt."""

    def __init__(self, filepath=None):
        self.filepath = filepath or KNOWLEDGE_BASE_FILE
        self.raw_text = ""
        self.sentences = []
        self.processed_sentences = []

    def load(self):
        """Load the knowledge base file and preprocess it."""
        if not os.path.exists(self.filepath):
            print(f"[ERROR] Knowledge base file not found: {self.filepath}")
            print("Please ensure 'travel_data.txt' exists in the project directory.")
            sys.exit(1)

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.raw_text = f.read()
        except Exception as e:
            print(f"[ERROR] Failed to read knowledge base: {e}")
            sys.exit(1)

        # Split into sentences (each line is a separate entry)
        self.sentences = [
            line.strip()
            for line in self.raw_text.split("\n")
            if line.strip()
        ]

        # Preprocess each sentence for matching
        self.processed_sentences = [
            self._preprocess(sentence) for sentence in self.sentences
        ]

        return self

    def _preprocess(self, text):
        """Preprocess text: lowercase, remove punctuation, normalize whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_sentence_count(self):
        """Return the number of sentences in the knowledge base."""
        return len(self.sentences)


# ═══════════════════════════════════════════════════════════════
# Query Processing Module
# ═══════════════════════════════════════════════════════════════

class QueryProcessor:
    """Processes and normalizes user queries for retrieval."""

    def process(self, query):
        """
        Process a user query:
        - Convert to lowercase
        - Remove punctuation
        - Extract meaningful keywords (remove stop words)
        """
        # Lowercase and remove punctuation
        query_clean = query.lower()
        query_clean = query_clean.translate(str.maketrans("", "", string.punctuation))
        query_clean = re.sub(r"\s+", " ", query_clean).strip()

        # Extract keywords (remove stop words)
        words = query_clean.split()
        keywords = [w for w in words if w not in STOP_WORDS]

        return {
            "original": query,
            "cleaned": query_clean,
            "keywords": keywords,
            "all_words": words,
        }


# ═══════════════════════════════════════════════════════════════
# Retrieval Module (TF-IDF Keyword Matching)
# ═══════════════════════════════════════════════════════════════

class Retriever:
    """
    Retrieves relevant sentences from the knowledge base
    using TF-IDF based keyword matching.
    """

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self._build_idf()

    def _build_idf(self):
        """Build IDF (Inverse Document Frequency) scores for all terms."""
        num_docs = len(self.kb.processed_sentences)
        doc_freq = Counter()

        for sentence in self.kb.processed_sentences:
            unique_words = set(sentence.split())
            for word in unique_words:
                doc_freq[word] += 1

        # Calculate IDF: log(N / df) + 1
        self.idf = {}
        for word, df in doc_freq.items():
            self.idf[word] = math.log(num_docs / df) + 1

    def _compute_tf(self, text):
        """Compute term frequency for a given text."""
        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return {}
        tf = Counter(words)
        return {word: count / word_count for word, count in tf.items()}

    def _score_sentence(self, sentence_idx, processed_query):
        """
        Score a sentence against the query using TF-IDF similarity.
        Also gives bonus weight for exact keyword matches.
        """
        sentence = self.kb.processed_sentences[sentence_idx]
        keywords = processed_query["keywords"]
        all_words = processed_query["all_words"]

        if not keywords and not all_words:
            return 0.0

        # Use keywords if available, otherwise fall back to all words
        query_terms = keywords if keywords else all_words

        # TF-IDF score
        sentence_tf = self._compute_tf(sentence)
        tfidf_score = 0.0

        for term in query_terms:
            if term in sentence_tf:
                tf = sentence_tf[term]
                idf = self.idf.get(term, 1.0)
                tfidf_score += tf * idf

        # Bonus: count how many unique query keywords appear in the sentence
        sentence_words = set(sentence.split())
        keyword_matches = sum(1 for kw in query_terms if kw in sentence_words)
        match_ratio = keyword_matches / len(query_terms) if query_terms else 0

        # Combined score: TF-IDF + keyword coverage bonus
        combined_score = tfidf_score + (match_ratio * 2.0)

        return combined_score

    def retrieve(self, processed_query, top_k=None):
        """
        Retrieve the top-K most relevant sentences from the knowledge base.

        Args:
            processed_query: Output from QueryProcessor.process()
            top_k: Number of results to return (default from config)

        Returns:
            List of (sentence, score) tuples sorted by relevance
        """
        top_k = top_k or TOP_K_RESULTS

        # Score all sentences
        scored = []
        for idx in range(len(self.kb.sentences)):
            score = self._score_sentence(idx, processed_query)
            if score >= MIN_RELEVANCE_SCORE:
                scored.append((self.kb.sentences[idx], score))

        # Sort by score (highest first) and return top-K
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ═══════════════════════════════════════════════════════════════
# Prompt Construction Module
# ═══════════════════════════════════════════════════════════════

class PromptBuilder:
    """Constructs prompts for the LLM by combining retrieved context with the user query."""

    def build(self, query, retrieved_results):
        """
        Build a structured prompt for the LLM.

        Args:
            query: Original user query string
            retrieved_results: List of (sentence, score) tuples from retriever

        Returns:
            Formatted prompt string
        """
        # Combine retrieved sentences into context block
        if retrieved_results:
            context = "\n".join(
                f"• {sentence}" for sentence, _ in retrieved_results
            )
        else:
            context = "(No specific information found in the knowledge base for this query.)"

        # Fill the prompt template
        prompt = QUERY_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        return prompt


# ═══════════════════════════════════════════════════════════════
# Gemini API Client Module
# ═══════════════════════════════════════════════════════════════

class GeminiClient:
    """Manages communication with the Google Gemini API."""

    def __init__(self):
        self.model = None
        self._initialized = False

    def initialize(self):
        """Initialize the Gemini API client."""
        api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")

        if not api_key:
            print("\n" + "=" * 60)
            print("  [!] GEMINI API KEY NOT FOUND")
            print("=" * 60)
            print("\nTo use this application, you need a Google Gemini API key.")
            print("\nSetup instructions:")
            print("  1. Go to https://aistudio.google.com/apikey")
            print("  2. Create a free API key")
            print("  3. Set it as an environment variable:")
            print('     Windows:  set GEMINI_API_KEY=your_key_here')
            print('     Linux:    export GEMINI_API_KEY=your_key_here')
            print("  Or create a .env file with: GEMINI_API_KEY=your_key_here")
            print("=" * 60)
            return False

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                generation_config=GENERATION_CONFIG,
                system_instruction=SYSTEM_PROMPT,
            )
            self._initialized = True
            return True

        except ImportError:
            print("\n[ERROR] google-generativeai package not installed.")
            print("Run: pip install google-generativeai")
            return False
        except Exception as e:
            print(f"\n[ERROR] Failed to initialize Gemini API: {e}")
            return False

    def generate(self, prompt):
        """
        Send a prompt to the Gemini API and return the response.

        Args:
            prompt: The constructed prompt string

        Returns:
            Generated text response, or error message
        """
        if not self._initialized:
            return "[ERROR] Gemini API is not initialized. Please set your API key."

        try:
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                return "[ERROR] API rate limit reached. Please wait a moment and try again."
            elif "api_key" in error_msg.lower() or "invalid" in error_msg.lower():
                return "[ERROR] Invalid API key. Please check your GEMINI_API_KEY."
            else:
                return f"[ERROR] API request failed: {error_msg}"


# ═══════════════════════════════════════════════════════════════
# RAG Engine (Orchestrator)
# ═══════════════════════════════════════════════════════════════

class RAGEngine:
    """
    Main RAG engine that orchestrates the entire pipeline:
    Query → Retrieval → Prompt Construction → LLM Generation
    """

    def __init__(self):
        self.kb = KnowledgeBase()
        self.query_processor = QueryProcessor()
        self.retriever = None
        self.prompt_builder = PromptBuilder()
        self.gemini_client = GeminiClient()

    def initialize(self):
        """Initialize all components of the RAG engine."""
        # Load knowledge base
        print("[*] Loading travel knowledge base...")
        self.kb.load()
        print(f"   [OK] Loaded {self.kb.get_sentence_count()} travel entries")

        # Build retriever
        print("[*] Building retrieval index...")
        self.retriever = Retriever(self.kb)
        print("   [OK] Retrieval index ready")

        # Initialize Gemini API
        print("[*] Connecting to Gemini AI...")
        api_ready = self.gemini_client.initialize()
        if api_ready:
            print("   [OK] Gemini AI connected")
        else:
            print("   [!!] Running in offline mode (retrieval only)")

        return api_ready

    def process_query(self, query):
        """
        Process a user query through the full RAG pipeline.

        Args:
            query: User's travel question

        Returns:
            dict with 'response', 'retrieved_count', and 'retrieved_context'
        """
        # Step 1: Process query
        processed = self.query_processor.process(query)

        # Step 2: Retrieve relevant context
        results = self.retriever.retrieve(processed)

        # Step 3: Build prompt
        prompt = self.prompt_builder.build(query, results)

        # Step 4: Generate response
        response = self.gemini_client.generate(prompt)

        return {
            "response": response,
            "retrieved_count": len(results),
            "retrieved_context": [sent for sent, _ in results],
        }


# ═══════════════════════════════════════════════════════════════
# Interactive CLI
# ═══════════════════════════════════════════════════════════════

def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 62)
    print("  Smart-Travel RAG Itinerary Generator")
    print("  Your AI Travel Guide for Jammu & Kashmir")
    print("=" * 62)
    print()


def print_help():
    """Print help information."""
    print("\n[?] Available Commands:")
    print("  - Type your travel question to get AI-powered recommendations")
    print("  - 'help'    -- Show this help message")
    print("  - 'info'    -- Show knowledge base information")
    print("  - 'exit'    -- Exit the program")
    print()
    print("[*] Example Queries:")
    print('  - "Plan a 2-day trip in Kashmir"')
    print('  - "Suggest hidden places in Kashmir"')
    print('  - "What is the best time to visit Gulmarg?"')
    print('  - "Tell me about Kashmiri food and cuisine"')
    print('  - "Plan a 5-day honeymoon itinerary in Kashmir"')
    print()


def run_cli():
    """Run the interactive command-line interface."""
    print_banner()

    # Initialize the RAG engine
    engine = RAGEngine()
    api_available = engine.initialize()

    print()
    print_help()

    # Main interaction loop
    while True:
        try:
            print("-" * 62)
            query = input("\n>> Your travel query: ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ("exit", "quit", "q", "bye"):
                print("\n>> Thank you for using Smart-Travel RAG!")
                print("   Have a wonderful trip!\n")
                break

            if query.lower() == "help":
                print_help()
                continue

            if query.lower() == "info":
                print(f"\n[i] Knowledge Base: {engine.kb.get_sentence_count()} travel entries loaded")
                print(f"   Source: {engine.kb.filepath}")
                print(f"   Model: {MODEL_NAME}")
                print(f"   Top-K retrieval: {TOP_K_RESULTS}")
                continue

            if not api_available:
                print("\n[!!] Gemini API is not available. Showing retrieved context only.\n")
                processed = engine.query_processor.process(query)
                results = engine.retriever.retrieve(processed)
                if results:
                    print(f"[*] Retrieved {len(results)} relevant entries:\n")
                    for i, (sentence, score) in enumerate(results, 1):
                        print(f"  {i}. {sentence}")
                        print(f"     (relevance: {score:.2f})")
                        print()
                else:
                    print("[x] No relevant information found for your query.")
                continue

            # Process the query through the RAG pipeline
            print("\n[...] Searching knowledge base and generating response...\n")
            result = engine.process_query(query)

            # Display results
            print(f"[*] Retrieved {result['retrieved_count']} relevant entries from knowledge base")
            print()
            print("--- AI Generated Response ---")
            print("-" * 62)
            print(result["response"])
            print("-" * 62)

        except KeyboardInterrupt:
            print("\n\n>> Goodbye! Happy travels!\n")
            break
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")
            print("Please try again with a different query.\n")


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_cli()
