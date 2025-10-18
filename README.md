# Nile eCommerce Review Prediction

## Overview

This project predicts which customers of "Nile," a major South American eCommerce platform, are most likely to leave positive reviews. The aim is to optimize targeting strategies for improving customer engagement and operational efficiency using machine learning.

## Features

- Utilizes a multi-table dataset (customer, order, payment, etc.)
- Applies the CRISP-DM data science methodology
- Exploratory Data Analysis (EDA) to detect key patterns
- Built, tuned & compared Random Forest and Gradient Boosted Decision Tree (GBDT) models
- Achieved an F1 score of 0.86 in predicting positive reviews
- Visualized insights in Tableau and presented actionable recommendations

## Data

Eight tables: customer reviews, order details, payments, products, and more. Data cleaning, feature engineering, and encoding operations performed to enhance predictive capability.

## Methodology

- **CRISP-DM Framework**:
  - Data understanding and preparation
  - EDA for trends, outliers, and feature significance
  - Feature engineering (delivery times, payment types, demographic info, etc.)
  - Model training with class imbalance handling
  - Random search for hyperparameter optimisation
  - Metrics: F1, precision, recall, accuracy

## Results

- Random Forest outperformed GBDT (macro F1: 0.86 on positive reviews)
- Influential features: overdue delivery days, payment amounts, order frequency
- Visualizations highlight review patterns by region and time

## Recommendations

- Improve delivery reliability for better customer reviews
- Streamline overlapping product categories for clearer analytics
- Provide targeted incentives to customers likely to leave positive reviews
- Continuously update and retrain model with new data

## Deployment

Model can be integrated via microservices, enabling real-time analytics and automated outreach to high-potential reviewers.

## Project Structure

