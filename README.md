# Hate Speech Classification

This project focuses on automatically detecting hate speech and offensive language in social media text using machine learning. The goal is to build a system that can identify harmful or abusive content to help maintain a safer online environment.

The project covers the complete end-to-end machine learning workflow — from data collection and preprocessing to model training, experiment tracking, and deployment. It integrates modern MLOps tools like **MLflow** for experiment tracking, **DVC** for data and model versioning, and **FastAPI** for serving the trained model as an API.

By combining natural language processing (NLP) techniques and machine learning algorithms, this project provides an efficient pipeline to classify text into categories such as *Hate Speech*, *Offensive Language*, or *Neutral*.



## Problem Statement

Social media platforms receive millions of comments and posts every day. Among them, some contain hate speech or offensive language that can harm individuals or communities. Manually monitoring and filtering such content is not practical due to the large volume of data.

This project aims to build an automated machine learning model that can detect and classify text as **Hate Speech**, **Offensive Language**, or **Neutral**. The goal is to support content moderation systems by accurately identifying harmful language and helping create a safer online space.



##  Project Architecture

The project follows a complete end-to-end machine learning workflow. It starts with collecting and cleaning the data, then moves on to training and evaluating the model. After that, the model is tracked, versioned, and finally deployed using modern MLOps tools.

Below is a simple view of the pipeline:

1. **Data Collection:** Gather hate speech and offensive language data.
2. **Data Preprocessing:** Clean and prepare text for modeling.
3. **Model Training:** Train machine learning models on the processed data.
4. **Model Evaluation:** Check how well the model performs using metrics like F1-score.
5. **Experiment Tracking:** Use MLflow to track experiments and results.
6. **Version Control:** Use DVC to manage datasets and model versions.
7. **Deployment:** Deploy the final model using FastAPI for real-time predictions.

This setup makes the workflow easy to reproduce, monitor, and improve over time.

##  Model and Techniques Used

The project uses natural language processing (NLP) and machine learning to classify text. Here’s what we do:

- **Text Preprocessing:** Clean the text by removing unnecessary characters, converting to lowercase, and removing stopwords.
- **Feature Extraction:** Convert text into numbers that the model can understand, using techniques like word2vec.
- **Model:** Train a machine learning model (e.g., RNN/LSTM) on the processed text.
- **Evaluation:** Measure the model's performance using metrics like F1-score, Precision, and Recall to make sure it correctly identifies hate speech and offensive language.


## 📁 Project Structure

Here’s how the project files and folders are organized:

│
├── data/ # Raw and processed data (created during data ingestion)
├── artifacts/ # Stores intermediate outputs like validation reports, transformed data, and trained models
├── src/ # Source code for all components
│ ├── components/ # Data ingestion, transformation, model training, and evaluation modules
│ ├── configuration/ # Configuration files (e.g., DB connections)
│ ├── constant/ # Constants used across the project
│ ├── entity/ # Data classes for configuration and artifacts
│ ├── exception/ # Custom exception handling
│ ├── logger/ # Logging setup
│ ├── pipeline/ # Training and prediction pipelines
│ ├── ml/ # Model definitions
│ └── utils/ # Utility functions
│
├── tests/ # Unit and integration tests
│ ├── unit/
│ └── integration/
│
├── experiments/ # Jupyter notebooks for experiments
├── .github/workflows/ # GitHub Actions workflows
├── Dockerfile # Docker configuration
├── app.py # FastAPI application for deployment
├── demo.py # Script for testing or demoing the model
├── requirements.txt # Project dependencies
├── requirements_dev.txt # Dev dependencies (testing, linting)
├── setup.py # Optional packaging setup
├── dvc.yaml # DVC pipeline configuration
├── params.yaml # Parameters for pipeline stages
├── README.md # Project documentation
├── .gitignore # Files/folders to ignore in Git
└── .env # Environment variables