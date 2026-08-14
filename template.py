import os
from pathlib import Path
import logging

# logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'Rag-Business-Assistant'

list_of_files = [
    # GitHub / CI
    ".github/workflows/ci.yml",

    # data/
    "data/raw/sec/.gitkeep",
    "data/raw/nhtsa/.gitkeep",
    "data/raw/honda_ir/.gitkeep",
    "data/processed/.gitkeep",
    "data/chunks/.gitkeep",
    "data/embeddings/.gitkeep",
    "data/structured/.gitkeep",
    "data/evaluation/eval_set.jsonl",

    # src/ingestion
    "src/__init__.py",
    "src/ingestion/__init__.py",
    "src/ingestion/sec_client.py",
    "src/ingestion/nhtsa_client.py",
    "src/ingestion/honda_ir_client.py",

    # src/processing
    "src/processing/__init__.py",
    "src/processing/parse_pdf.py",
    "src/processing/clean.py",
    "src/processing/chunk.py",

    # src/embeddings
    "src/embeddings/__init__.py",
    "src/embeddings/embed.py",

    # src/vectorstore
    "src/vectorstore/__init__.py",
    "src/vectorstore/qdrant_store.py",

    # src/retrieval
    "src/retrieval/__init__.py",
    "src/retrieval/retriever.py",
    "src/retrieval/reranker.py",
    "src/retrieval/rag_answer.py",

    # src/structured
    "src/structured/__init__.py",
    "src/structured/schema.sql",
    "src/structured/load_data.py",
    "src/structured/sql_tool.py",

    # src/agents
    "src/agents/__init__.py",
    "src/agents/graph.py",
    "src/agents/tools.py",

    # src/mcp
    "src/mcp/__init__.py",
    "src/mcp/server.py",
    "src/mcp/client.py",

    # src/api
    "src/api/__init__.py",
    "src/api/main.py",
    "src/api/schemas.py",
    "src/api/config.py",

    # src/eval
    "src/eval/__init__.py",
    "src/eval/run_ragas.py",
    "src/eval/metrics.py",

    # frontend/
    "frontend/app.py",

    # notebooks/
    "notebooks/exploration.ipynb",

    # tests/
    "tests/__init__.py",
    "tests/test_retrieval.py",
    "tests/test_sql_tool.py",
    "tests/test_agent.py",

    # configs/
    "configs/settings.yaml",

    # scripts/
    "scripts/run_ingestion.sh",
    "scripts/build_index.sh",
    "scripts/run_eval.sh",

    # docker/
    "docker/Dockerfile.backend",
    "docker/Dockerfile.frontend",
    "docker/docker-compose.yml",

    # root-level files
    ".env.example",
    ".gitignore",
    "requirements.txt",
    "README.md",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")