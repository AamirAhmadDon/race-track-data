CREATE DATABASE IF NOT EXISTS motosport_DB;
USE motosport_DB;
CREATE TABLE races(
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(100),
    date DATE,
    season VARCHAR(20)
);
CREATE TABLE drivers(
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    nationality VARCHAR(50),
    team VARCHAR(100)
);
CREATE TABLE results(
    id INT AUTO_INCREMENT PRIMARY KEY,
    race_id INT,
    driver_id INT,
    position INT,
    lap_time DECIMAL(6,3),
    FOREIGN KEY (race_id) REFERENCES races(id) ON DELETE CASCADE,
    FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE CASCADE
    );
CREATE TABLE model_metrics(
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_name VARCHAR(100),
    accuracy DECIMAL(5,4),
    loss DECIMAL(8,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);