# 🎬 AI-Powered Recommendation System (RAG-Enabled, Production-Ready)

A production-style recommendation system that combines **ensemble machine learning models** with **Retrieval-Augmented Generation (RAG)** to deliver context-aware movie recommendations.

Built with a focus on **system design, scalability, and real-world ML deployment practices**.

---

## 🚀 Project Overview

This project simulates a real-world recommendation engine similar to systems used by platforms like Netflix or Amazon.

It combines:

* Collaborative filtering using ML models
* Content-based retrieval using embeddings and vector search
* API-first architecture using FastAPI
* Containerized deployment using Docker

The system predicts user preferences and enhances recommendations with semantic context from similar items.

---

## 🧠 Architecture (High-Level)

User Request
→ FastAPI (REST API)
→ Recommendation Service (ML Models)
→ Top-N Predictions
→ Retrieval Service (FAISS + Embeddings)
→ Context-Aware Results + Explanations
→ Response

---

## ⚙️ Tech Stack

### Backend & API

* FastAPI
* Uvicorn

### Machine Learning

* Scikit-learn (Ridge, Random Forest, Gradient Boosting)
* Ensemble Learning

### Data Processing

* Pandas
* NumPy

### Retrieval (RAG)

* Sentence Transformers
* FAISS (vector similarity search)

### Deployment

* Docker
* AWS-ready architecture (EC2 compatible)

### Testing & Performance

* Locust (load testing)

---

## 📊 Features

### ✅ Ensemble Recommendation Engine

* Combines Ridge Regression, Random Forest, and Gradient Boosting
* Improves prediction accuracy over single-model approaches

---

### ✅ Context-Aware Recommendations (RAG)

* Uses semantic similarity (FAISS + embeddings)
* Enhances recommendations with similar movie context
* Generates explanations for recommendations

---

### ✅ Scalable API System

* Built using FastAPI
* Supports real-time recommendation queries
* Designed for concurrent usage scenarios

---

### ✅ Load Testing & Performance Analysis

* Simulated concurrent users using Locust
* Identified bottlenecks in the retrieval pipeline
* Measured latency and throughput under load

---

## 📈 Sample API Endpoints

### 🔹 Get Recommendations

GET /recommend?user_id=50&k=5

Example response:

```json
{
  "user_id": 50,
  "recommendations": [
    {
      "movie_id": 1653,
      "title": "Entertaining Angels",
      "predicted_rating": 4.31,
      "similar_context": ["One Fine Day", "Eighth Day"],
      "explanation": "Recommended based on predicted preference and semantic similarity"
    }
  ]
}
```

---

### 🔹 Semantic Search

GET /similar?query=toy story&k=5

---

## 📂 Project Structure

reco-rag-system/
│
├── app/                  # API and services
├── artifacts/            # Models, features, vector index
├── data/                 # Raw dataset
├── loadtest/             # Locust performance testing
├── scripts/              # Data processing and training
├── Dockerfile
├── requirements.txt
└── README.md

---

## 🧪 Load Testing Results

* Tested using Locust
* Achieved ~5–6 requests/sec on a local setup
* Identified performance bottleneck in the embedding-based retrieval layer

Key observation:

Repeated embedding computation increases latency (~10–12 seconds per recommendation request under load).

---

## 🧠 Key Learnings

* Trade-offs between accuracy and latency in recommendation systems
* Importance of feature engineering for user and item signals
* Challenges in deploying ML systems across environments
* Need for caching and precomputation in RAG pipelines
* Handling model compatibility issues in production

---

## 🐳 Running with Docker

Build the image:

docker build -t reco-rag-app .

Run the container:

docker run -p 8000:8000 reco-rag-app

Then open:

http://localhost:8000/docs

---

## 📌 Future Improvements

* Precompute embeddings to reduce inference latency
* Add caching layer (e.g., Redis)
* Deploy on AWS EC2 with autoscaling
* Improve personalization using user behavior features
* Add monitoring (Prometheus, Grafana)

---

## 💡 Why this project matters

This project demonstrates:

* End-to-end ML system design
* Integration of ML models with RAG pipelines
* Production-ready API development
* Performance evaluation under load

---

## 👤 Author

Built as a production-style ML engineering project to demonstrate real-world system design and deployment capabilities.
