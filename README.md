# Titanic Survival Prediction using Naive Bayes

## Overview
This project analyzes the Titanic dataset and builds a **Naive Bayes classifier** to predict survival based on key features such as age, class, and fare. The dataset is preprocessed, categorical variables are encoded, and missing values are handled before training the model.

## Dataset
The dataset is loaded from the Seaborn library and contains the following columns:
- `survived` - Target variable (0 = No, 1 = Yes)
- `pclass` - Passenger class (1st, 2nd, 3rd)
- `sex` - Gender
- `age` - Passenger age
- `fare` - Ticket fare
- `class` - Ticket class (First, Second, Third)

## Data Preprocessing
- Removed irrelevant columns (`deck`, `sibsp`, `parch`, `embark_town`, `embarked`, `alive`, `alone`, `adult_male`, `who`).
- Filled missing `age` values with the mean age.
- Encoded categorical variables (`sex` and `class`) using `LabelEncoder`.
- Converted `age` and `fare` to integers.

## Model Training
- Split the dataset into training (80%) and testing (20%) sets.
- Trained a **Gaussian Naive Bayes** model on the dataset.
- Evaluated the model using accuracy score, confusion matrix, and classification report.

## Model Performance
- **Accuracy Score:** 75.4%
- **Confusion Matrix:**
  ```
  [[83 22]
   [22 52]]
  ```
- **Classification Report:**
  ```
              precision    recall  f1-score   support

           0       0.79      0.79      0.79       105
           1       0.70      0.70      0.70        74

    accuracy                           0.75       179
   macro avg       0.75      0.75      0.75       179
weighted avg       0.75      0.75      0.75       179
  ```

## Visualization
- A heatmap is generated to visualize the confusion matrix.

## How to Run the Project
1. Install dependencies:
   ```bash
   pip install numpy pandas seaborn scikit-learn matplotlib
   ```
2. Run the script in a Jupyter Notebook or Python environment.

## Conclusion
The **Naive Bayes classifier** achieved a **75.4% accuracy**, demonstrating a reasonable predictive capability for Titanic survival prediction. Further improvements can be made by feature engineering and trying other models.

---
### 📌 Author: Hamza Aslam
