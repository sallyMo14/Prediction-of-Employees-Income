# Prediction of Employees' Income

A machine learning project in a Jupyter Notebook that predicts whether an adult earns over or under a certain income threshold (e.g., >\$50K) using demographic and employment-related features.

---

##  Dataset Details

The project uses the UCI Adult dataset (also known as "Census Income Dataset"), which includes attributes such as age, education level, marital status, occupation, work hours, and more. The target variable represents whether an adult's income exceeds \$50K/year.
## Data Dictionary
- Age(numeric)
- Workclass (Categorical)
- fnlwgt (Numerical): How many population does this row represent from the sample
- Education (Ordinal):
- educational-num(Numerical)
- marital-status (Categorical)
- occupation (Categorical)
- relationship (Categorical)
- race (Categorical)
- gender (Categorical)
- capital-gain (Numerical): Income from investment sources
- capital-loss (Numerical)
- hours-per-week (Numerical)
- native-country (Numerical)

##  Workflow

- Check for null \ impossible data
- Exploratory data analysis (EDA)
- Feature preprocessing:
    - StanderdScaling for numerical features
    - OrdinalEncoder and scaler for ordinal features
    - OneHotEncoder for Categorical Data
-  Data Balancing  using **SMOTE**
-  Model training using:
   - LogisticRegression
   - RandomForestClassifier
   - DecisionTreeClassifier
   - KNeighborsClassifier
- Evaluation of each model using accuracy, precision, recall, F1-score, and confusion matrices
- Comparison of model performances, and select the **best** baseline model: **RandomForestClassifier**
- Extracting the **10** most important features
-  Feature Engineering:
  - Feature selection
  - Feature extraction
  - Dimensionality reduction using **PCA**
