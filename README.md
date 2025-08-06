# Data Mining on Mental Health Survey Data for Analyzing Depression Risk Among Students

## 1. Project Overview
This project analyzes a large-scale survey of university students' mental health to identify and predict risk factors for depression. Leveraging data mining and machine learning, we aim to provide actionable insights for early screening and intervention in educational settings.

## 2. Dataset
- **Name:** Student Depression Dataset  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
- **Size:** ~27,000 samples, 18 features
- **Target:** Depression (binary: Yes/No)
- **Attribute groups:**
  - Demographics: Age, Gender, City
  - Academic: CGPA, Academic Pressure, Study Satisfaction
  - Lifestyle & Wellbeing: Sleep Duration, Dietary Habits, Work/Study Hours, Job Satisfaction
  - Additional: Financial Stress, Family Mental History, Suicidal Thoughts

## 3. Data Preprocessing
- Column name normalization
- Handling missing/invalid values (e.g., impute financial_stress with median)
- Grouping rare categories (e.g., City → Others)
- Encoding categorical variables (Label, One-Hot, Binary)
- Removing irrelevant/low-variance features
- Feature engineering (e.g., stress_interaction, sleep_adequate)  

## 4. Exploratory Data Analysis (EDA)
- Distribution of depression status (58.5% depressed)
- Analysis by gender, age group, CGPA, financial_stress, dietary_habits, suicidal_thoughts
- Visualizations: pie, histograms, ratio plot, bar plots, correlation heatmap

## 5. Machine Learning Models
- **Logistic Regression:** Interpretable baseline, best overall accuracy (0.84)
- **Decision Tree:** Visual decision-making, accuracy ~0.82
- **Random Forest:** Handles complexity, accuracy ~0.83
- **XGBoost:** Boosted accuracy, best recall for depression class
- **Naive Bayes:** Simple, high recall for positive class
- **Workflow:** Data split, model training, evaluation (accuracy, F1, ROC-AUC)

## 6. Association Rule Mining
- Apriori algorithm to discover rules (e.g., "academic_pressure=Medium" + "suicidal_thoughts" → depression)
- Example: Medium academic pressure & unhealthy diet greatly increase depression risk

## 7. Clustering
- **KMeans:** Best with k=2, splits into two main student profiles
- **Agglomerative:** Less effective due to data structure
- **Purpose:** Identify hidden groups for further analysis

## 8. Results & Evaluation
- Best models: Logistic Regression & Random Forest (Accuracy >83%)
- Top features: Suicidal thoughts, academic pressure, financial stress, age
- Association rules and clustering supplement ML findings

## 9. Limitations & Future Work
- Dataset imbalance in some categories (e.g., gender, city)
- Mostly categorical variables, limited granularity
- Did not apply deep learning or time-series
- Future: Larger, richer datasets, deep models, sentiment analysis from text, real-time/longitudinal data

## 10. How to Run
- Requirements: Python 3.x, pandas, scikit-learn, xgboost, mlxtend, matplotlib, seaborn
- Example usage:
    ```bash
    python preprocess.py
    python train_models.py
    python run_apriori.py
    ```
- Jupyter notebooks available for step-by-step analysis

## 11. Team & Credits
- **Members:**
    - [Nguyễn Vĩ Khang](https://github.com/khangvbeauty): Feature engineering, modeling, documentation
    - Trương Thái Bảo: EDA & visualization, modeling, writing
    - [Thạch Nhựt Hào](https://github.com/haothach): Data cleaning, modeling, reporting
- **References:**
    - Main dataset: [Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
    - Academic papers and background (see report)
