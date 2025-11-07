# Hate Speech Classification

This project focuses on automatically detecting hate speech and offensive language in social media text using machine learning. The goal is to build a system that can identify harmful or abusive content to help maintain a safer online environment.

The project covers the complete end-to-end machine learning workflow — from data collection and preprocessing to model training, experiment tracking, and deployment. It integrates modern MLOps tools like **MLflow** for experiment tracking, **DVC** for data and model versioning, and **FastAPI** for serving the trained model as an API.

By combining natural language processing (NLP) techniques and machine learning algorithms, this project provides an efficient pipeline to classify text into categories such as *Hate Speech*, *Offensive Language*, or *Neutral*.



## Problem Statement

Social media platforms receive millions of comments and posts every day. Among them, some contain hate speech or offensive language that can harm individuals or communities. Manually monitoring and filtering such content is not practical due to the large volume of data.

This project aims to build an automated machine learning model that can detect and classify text as **Hate Speech**, **Offensive Language**, or **Neutral**. The goal is to support content moderation systems by accurately identifying harmful language and helping create a safer online space.



## ⚙️ Project Architecture

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


