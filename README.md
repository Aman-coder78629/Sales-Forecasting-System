# Sales-Forecasting-System

# 📈 Sales Forecasting System

> **An End-to-End Sales Forecasting & Demand Intelligence System using Time Series Analysis, Machine Learning, Anomaly Detection, Product Segmentation, and Interactive Dashboard Deployment with Streamlit.**

# 📌 Project Overview

Inventory management is one of the most critical challenges faced by retail and e-commerce businesses. Overstocking products increases storage costs and ties up working capital, while understocking leads to lost sales and dissatisfied customers. Accurate sales forecasting enables businesses to maintain the right inventory levels, optimize supply chains, and improve customer satisfaction.

This project presents a complete **Sales Forecasting & Demand Intelligence System** developed using historical Superstore sales data. The system combines **Time Series Analysis**, **Machine Learning**, **Anomaly Detection**, **Product Demand Segmentation**, and **Business Intelligence Visualization** to help organizations forecast future sales and make data-driven inventory decisions.

The project was developed as part of a Data Science Internship and simulates a real-world retail analytics solution similar to those used by organizations such as Amazon, Walmart, Flipkart, and D-Mart.

---

# 🎯 Project Objectives

The primary objectives of this project are:

- Analyze historical sales performance.
- Identify sales trends and seasonality.
- Forecast future sales for the next three months.
- Compare multiple forecasting algorithms.
- Detect unusual sales spikes and drops.
- Segment products based on demand behavior.
- Build an interactive Streamlit dashboard for business users.
- Generate business recommendations based on data insights.

# 📊 Dataset

### Primary Dataset

**Superstore Sales Dataset**

The dataset contains approximately four years of retail transaction data, including:

- Order Date
- Ship Date
- Sales
- Profit
- Quantity
- Discount
- Product Category
- Sub-Category
- Customer Segment
- Region
- State
- City

Dataset Source

https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting

---

### Supplementary Dataset

Video Game Sales Dataset

Used for understanding multi-source data analysis and anomaly detection techniques.

Dataset Source

https://www.kaggle.com/datasets/gregorut/videogamesales

---

# 🛠 Technologies Used

## Programming Language

- Python 3.x

## Development Environment

- Jupyter Notebook
- Visual Studio Code

## Libraries

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Time Series Analysis

- Statsmodels
- Prophet

### Machine Learning

- XGBoost
- Scikit-learn

### Clustering

- K-Means
- PCA

### Dashboard

- Streamlit

### Version Control

- Git
- GitHub

---

# 📌 Project Workflow

The project is divided into eight major tasks.

---

# Task 1 — Data Loading & Exploratory Data Analysis

The first phase focuses on understanding the dataset.

Activities performed include:

- Loading CSV files
- Parsing datetime columns
- Handling missing values
- Removing duplicate records
- Feature Engineering
- Weekly aggregation
- Monthly aggregation

Business Questions Answered

- Which category generates maximum revenue?
- Which region performs best?
- Average shipping duration.
- Seasonal sales trends.

---

# Task 2 — Time Series Analysis

Monthly sales data was converted into a time series.

The following analyses were performed:

- Monthly Trend Analysis
- Seasonal Decomposition
- Trend Component
- Seasonal Component
- Residual Analysis

Stationarity was checked using

- Augmented Dickey-Fuller Test

If necessary, differencing was applied before forecasting.

---

# Task 3 — Sales Forecasting

Three different forecasting models were implemented and compared.

## Model 1

### SARIMA

Statistical forecasting model capable of modeling trend and seasonality.

Features

- Seasonal forecasting
- Confidence intervals
- Time series decomposition

---

## Model 2

### Facebook Prophet

An industry-standard forecasting framework developed by Meta.

Advantages

- Handles seasonality automatically
- Handles holidays
- Easy forecasting

---

## Model 3

### XGBoost Regressor

Machine Learning approach using lag features.

Features engineered

- Lag 1
- Lag 2
- Lag 3
- Rolling Mean
- Month
- Quarter
- Season

---

## Model Evaluation

Models were evaluated using

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

The model with the lowest prediction error was selected for deployment.

---

# Task 4 — Category & Region Forecasting

The selected forecasting model was applied independently to

### Product Categories

- Furniture
- Technology
- Office Supplies

### Regions

- West
- East

Separate three-month forecasts were generated for each segment.

Comparison graphs were produced to identify the fastest-growing business segment.

---

# Task 5 — Sales Anomaly Detection

Unusual sales patterns were identified using two different approaches.

## Isolation Forest

Machine Learning-based anomaly detection.

Detects

- Unexpected spikes
- Sudden drops
- Rare sales events

---

## Z-Score Detection

Statistical anomaly detection.

Flags weeks where sales differ significantly from historical averages.

The results from both methods were compared to improve confidence in anomaly identification.

---

# Task 6 — Product Demand Segmentation

Products were grouped into demand segments using K-Means Clustering.

Features used

- Total Sales Volume
- Growth Rate
- Monthly Volatility
- Average Order Value

The Elbow Method determined the optimal number of clusters.

PCA was used to visualize clusters in two dimensions.

Demand segments include

- High Volume Stable Demand
- Growing Demand
- Low Volume High Volatility
- Declining Demand

Each cluster was associated with inventory management recommendations.

---

# Task 7 — Interactive Streamlit Dashboard

A fully interactive dashboard was developed using Streamlit.

### Dashboard Features

## Sales Overview

- Total yearly sales
- Monthly sales trend
- Interactive filters
- Regional analysis

---

## Forecast Explorer

- Forecast by Category
- Forecast by Region
- Forecast Horizon Selection
- Model Accuracy Metrics

---

## Anomaly Report

- Weekly anomaly chart
- Detected anomaly table
- Sales spike visualization

---

## Product Segmentation

- Cluster visualization
- Demand group table
- Inventory recommendations

---

# Task 8 — Executive Business Report

A professional business report was prepared for senior management containing

- Executive Summary
- Forecast Results
- Business Insights
- Anomaly Analysis
- Product Segmentation
- Business Recommendations
- Project Limitations

---

# 📈 Results

The developed system successfully

- Forecasted future sales
- Detected abnormal sales patterns
- Identified seasonal demand
- Compared multiple forecasting techniques
- Segmented products into meaningful demand groups
- Built an interactive business dashboard

The solution enables organizations to

- Improve inventory planning
- Reduce stockouts
- Reduce excess inventory
- Increase operational efficiency
- Support strategic business decisions

---



# 📊 Future Scope

Possible future enhancements include

- LSTM-based Deep Learning Forecasting
- ARIMA Hyperparameter Optimization
- AutoML Integration
- Real-time Sales Forecasting
- Cloud Deployment (AWS/Azure/GCP)
- Inventory Optimization Module
- Sales Recommendation Engine
- REST API Deployment
- Power BI Integration

---

# 💼 Business Impact

This project demonstrates how Data Science can improve business performance by

- Predicting future demand
- Supporting procurement planning
- Improving inventory utilization
- Detecting unusual sales behavior
- Assisting supply chain management
- Reducing operational costs
- Increasing customer satisfaction

### Skills

- Python
- SQL
- Machine Learning
- Time Series Analysis
- XGBoost
- Streamlit
- Power BI
- Data Visualization



---

# ⭐ If you found this project helpful, consider giving it a Star on GitHub!
