This project is a Python + MySQL pipeline for managing motorsport data, extracting entities using SpaCy, RegEx, and Presidio, and training a simple PyTorch model to predict race outcomes.
📌 Features
✅ MySQL Database with structured tables for races, drivers, results, and ML metrics
✅ JSON-based configuration for DB credentials and regex patterns
✅ Entity Recognition with spaCy + Presidio + custom regex
✅ Error Handling for DB operations, file loading, and invalid configs
✅ PyTorch Model for basic predictive analytics
✅ Logging & Metrics stored in the database (model_metrics table)

🛠️ Tech Stack:
Python 3.9+
MySQL 9.x
Libraries:
SpaCy
presidio-analyzer
PyTorch
pandas
mysql-connector-python
