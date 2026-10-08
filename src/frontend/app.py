import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")
    company = data.get("company")

    try:
        resp = requests.post(
            f"{BACKEND_URL}/chat",
            json={"query": query, "company": company},
            timeout=300,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.exceptions.ConnectionError:
        return jsonify({"answer": "Error: Cannot connect to backend. Is the FastAPI server running on port 8000?", "sources": [], "tool_trace": []})
    except Exception as e:
        return jsonify({"answer": f"Error: {e}", "sources": [], "tool_trace": []})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    app.run(host="0.0.0.0", port=port, debug=False)
