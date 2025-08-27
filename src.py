import re
import json
import mysql.connector
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from presidio_analyzer import AnalyzerEngine, PatternRecognizer

#loading config
try:
    with open("pattrens.json", "r") as f:
        config = json.load(f)
    DB_CONFIG = config["DB_CONFIG"]
    regex_patterns = config["patterns"]
except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
    raise SystemExit(f"[CONFIG ERROR] Failed to load config: {e}")

#NLP & pii recognizer setup
nlp = spacy.load("en_core_web_sm")
analyzer = AnalyzerEngine()

for item in regex_patterns:
    try:
        label = item["label"]
        pattern = item["pattern"]
        recognizer = PatternRecognizer(supported_entity=label, patterns=[{"name": f"{label}_pattern", "regex": pattern}])
        analyzer.registry.add_recognizer(recognizer)
    except KeyError as e:
        print(f"[WARN] Skipping invalid pattern entry: {item} ({e})")

#database handler
class DatabaseHandler:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = mysql.connector.connect(**self.config)
            self.cursor = self.conn.cursor()
            print("[DB] Connected successfully.")
        except mysql.connector.Error as e:
            raise SystemExit(f"[DB ERROR] {e}")

    def insert(self, query, values):
        try:
            self.cursor.execute(query, values)
            self.conn.commit()
        except mysql.connector.Error as e:
            print(f"[DB INSERT ERROR] {e}")

    def close(self):
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()
        print("[DB] Connection closed.")

#pytorch model
class SimpleRacePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRacePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc2(self.relu(self.fc1(x))))

#main pipeline
def main():
    # DB connect
    db = DatabaseHandler(DB_CONFIG)
    db.connect()

    # Load CSV
    try:
        df = pd.read_csv("data/races.csv")
        print(f"[DATA] Loaded {len(df)} rows from CSV.")
    except FileNotFoundError:
        db.close()
        raise SystemExit("[ERROR] races.csv not found in data/ folder.")

    # Extract info with regex + NLP
    for _, row in df.iterrows():
        text = str(row.get("Race_Info", ""))
        doc = nlp(text)

        entities = analyzer.analyze(text=text, language="en")
        extracted = {item["entity_type"]: item["start"] for item in entities}

        # Insert into races table
        query = "INSERT INTO races (name, location, date, season) VALUES (%s, %s, %s, %s)"
        values = (row.get("Race_Name", "Unknown"), row.get("Location", "Unknown"),
                  row.get("Date", None), row.get("Season", None))
        db.insert(query, values)

    # Dummy training
    X = torch.rand(50, 10)
    y = torch.randint(0, 2, (50,))
    model = SimpleRacePredictor(input_size=10, hidden_size=16, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"[TRAIN] Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

    # Save metrics
    db.insert(
        "INSERT INTO model_metrics (experiment_name, accuracy, loss) VALUES (%s, %s, %s)",
        ("RacePredictor-v1", 0.92, float(loss.item()))
    )

    db.close()

if __name__ == "__main__":
    main()
