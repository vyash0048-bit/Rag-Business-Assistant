import sqlite3, os

def sql_query(query: str) -> str:
    """Run a read-only SQL SELECT against the business database and return rows as text."""
    q = query.strip().lower()
    if not q.startswith("select"):
        return "Error: only SELECT statements are allowed."
    conn = sqlite3.connect(os.environ.get("SQLITE_PATH", "data/structured/business.db"))
    try:
        rows = conn.execute(query).fetchall()
        cols = [d[0] for d in conn.execute(query).description]
        return "\n".join(str(dict(zip(cols, r))) for r in rows[:50])
    except Exception as e:
        return f"SQL error: {e}"
