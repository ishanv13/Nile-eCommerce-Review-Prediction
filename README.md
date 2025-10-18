# Nile eCommerce Review Prediction

## Project Summary

This repository features a predictive analytics project focused on the "Nile" eCommerce platform, aiming to forecast which customers are likely to leave positive reviews. By uncovering actionable insights from customer, order, and payment data, the project supports improved marketing and engagement strategies.

## Key Components

- Multi-table integration: Customer reviews, order details, payments, and more.
- Data pipeline: Cleaning, feature engineering, and exploratory analysis using the CRISP-DM framework.
- Modelling: Random Forest and Gradient Boosted Decision Tree (GBDT) algorithms evaluated for review prediction.
- Performance: Achieved an F1 score of 0.86 on positive review classification.
- Visualization: Tableau dashboards provide insights into review patterns and geographic trends.

## Workflow

1. **Data Preparation**  
   Cleaned and merged eight tables. Performed advanced feature engineering (delivery timing, payment type, frequency).

2. **Exploratory Data Analysis**  
   Analyzed feature relationships like the impact of delivery times and payment methods on review outcomes.

3. **Model Development**  
   - Compared Random Forest and GBDT.
   - Tuned hyperparameters and addressed class imbalance.
   - Validated results on test data.

4. **Insights and Reporting**  
   - Identified key factors affecting review positivity.
   - Presented findings to stakeholders via Tableau visualizations and technical presentations.

## Main Findings

- Reliable delivery and efficient payment handling strongly correlate with positive reviews.
- Targeting customers with timely deliveries and preferred payment methods increases positive feedback.
- Regular model retraining helps adapt to new trends.

## Usage

- Download or clone the repository from [GitHub](https://github.com/ishanv13/Nile-eCommerce-Review-Prediction/tree/main).
- Place the cleaned dataset files in the `/data` directory.
- Run `Cleaning and Merging Codebase.py` for initial data preparation.
- Execute `Modelling Code.py` to train and evaluate the predictive models.
- Explore EDA and modeling steps with the Jupyter notebooks in `/notebooks`.
- Generate and view business visualizations from the `/visualizations` directory.
- Review the project report (`Nile eCommerce Review Prediction Review.pdf`) for a comprehensive summary of findings and recommendations.

> **Note:** For best results, use a Python 3.x environment with all necessary packages installed (see script requirements).

## Project Structure
.
├── data/            # Source and cleaned datasets
├── notebooks/       # Jupyter notebooks for analytics and ML
├── models/          # Model training and evaluation scripts
└── visualizations/  # Tableau outputs and figures

## Team

Developed collaboratively as a university project by data science and analytics students.

## License

This project is open for educational and non-commercial use.
