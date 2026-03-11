"""
Smart-Travel RAG Itinerary Generator - Web Application
========================================================
Flask-based web interface for the RAG travel assistant.

Usage:
    python app.py
    Then open http://localhost:5000 in your browser.
"""

import os
import sys

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, render_template, request, jsonify
from main import RAGEngine

# ─────────────────────────────────────────────
# Flask Application Setup
# ─────────────────────────────────────────────

app = Flask(__name__)

# Initialize RAG engine (shared across requests)
print("\n>> Starting Smart-Travel RAG Web Server...\n")
rag_engine = RAGEngine()
api_available = rag_engine.initialize()
print()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main web interface."""
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """API endpoint to process a travel query."""
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Please enter a travel query."}), 400

        if not api_available:
            # Offline mode: return retrieved context only
            processed = rag_engine.query_processor.process(query)
            results = rag_engine.retriever.retrieve(processed)
            if results:
                context_text = "\n\n".join(
                    f"**{i}.** {sentence}" for i, (sentence, _) in enumerate(results, 1)
                )
                response_text = (
                    "⚠️ **Gemini API is not available.** "
                    "Showing retrieved knowledge base entries:\n\n"
                    + context_text
                )
            else:
                response_text = "No relevant information found in the knowledge base."

            return jsonify({
                "response": response_text,
                "retrieved_count": len(results),
                "api_available": False,
            })

        # Full RAG pipeline
        result = rag_engine.process_query(query)

        return jsonify({
            "response": result["response"],
            "retrieved_count": result["retrieved_count"],
            "api_available": True,
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/api/info", methods=["GET"])
def info():
    """Return knowledge base info."""
    return jsonify({
        "entries": rag_engine.kb.get_sentence_count(),
        "api_available": api_available,
        "model": "gemini-2.0-flash",
    })


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Smart-Travel RAG Web Server")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50)
    print()
    app.run(debug=False, host="0.0.0.0", port=5000)
