import sqlite3

def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT,
        result TEXT,
        probability REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def insert_prediction(video_name, result, probability):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (video_name, result, probability)
    VALUES (?, ?, ?)
    """, (video_name, result, probability))

    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    data = cursor.fetchall()

    conn.close()
    return data