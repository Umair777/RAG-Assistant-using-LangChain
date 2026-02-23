---
title: GenAI RAG Assistant
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---


# RAG Assistant Using LangChain

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mohdumair1996/GenAI)

A modular, production-aligned Retrieval-Augmented Generation (RAG) system built with LangChain and deployed via Docker.

---

## Overview

This project implements a structured RAG architecture designed for policy-driven enterprise and HR interaction use cases.  

It emphasizes reproducibility, modularity, and cloud-ready deployment.

---

## Architecture

User Query
│
▼
Retriever Layer
│
▼
Vector Database
│
▼
Context Assembly
│
▼
LLM Response Generation


The system separates ingestion, embedding, retrieval, and response generation into independent modules to enable experimentation and production hardening.

---

## Core Components

- Document ingestion pipeline
- Modular embedding interface
- Vector database integration
- Retriever abstraction
- Interactive chatbot interface
- Docker-based deployment
- Hugging Face Spaces integration

---

## Repository Structure

RAG-Assistant-using-LangChain/
│
├── Dockerfile
├── requirements.txt
├── README.md
├── data/
└── src/
├── app.py
├── AI_bot/
├── embeddings/
├── loaders/
├── retriever/
├── splitters/
├── vectordb/
└── pipeline/


---

## Local Development

### 1. Clone Repository

git clone <repo-url>
cd RAG-Assistant-using-LangChain


### 2. Create Virtual Environment



python3 -m venv .venv
source .venv/bin/activate


### 3. Install Dependencies



pip install -r requirements.txt


### 4. Run Application



python src/app.py


Application runs at:



http://localhost:7860


---

## Docker Execution

### Build



docker build -t rag-assistant .


### Run



docker run -p 7860:7860 rag-assistant


---

## Cloud Deployment

This project supports Docker-based deployment to Hugging Face Spaces.

Deployment branch: `RAG-deployment`

Steps:

1. Connect GitHub repository
2. Select Docker SDK
3. Choose CPU Basic hardware
4. Automatic build and deployment

---

## Versioning Strategy

- `main` — active development
- `RAG-deployment` — cloud-ready build
- Tagged releases — stable versions

---

## Future Roadmap

- Persistent vector storage
- Multi-user session management
- Authentication integration
- Evaluation metrics for RAG quality
- Production logging & observability
- SaaS-ready backend separation

---

## License

MIT License

Copyright (c) 2026 Umair Ashraf
