# Life Expectancy Data Analysis

## 📌 Project Overview

This project analyzes a global dataset on life expectancy provided by the World Health Organization (WHO). The main goal is to understand the key factors that influence life expectancy across different countries and years.

---

## 🎯 Objectives

- Explore trends and patterns in life expectancy across countries.
- Identify health, economic, and social indicators related to higher or lower life expectancy.
- Visualize relationships using statistical graphs and heatmaps.
- Build a machine learning model to predict life expectancy based on various features.

---

## 🧾 Dataset Information

**Source**: [Kaggle - Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)\
**File**: `life_expectancy.csv`

### Key Features:

- Country
- Year
- Status (Developed / Developing)
- Life expectancy
- Adult Mortality
- Alcohol Consumption
- Hepatitis B Immunization
- BMI
- GDP
- Schooling

---

## 🛠️ Tools & Libraries

- Python 3.x
- Pandas
- NumPy
- Seaborn / Matplotlib
- Scikit-learn
- Jupyter Notebook or VS Code

---

## 📁 Project Structure

```
life_expectancy_project/
├── data/
│   └── life_expectancy.csv
├── notebook/
│   └── 01_eda_visualization.ipynb
├── output/
│   └── charts/
├── README.md
```

---

## 📊 Steps Performed

1. **Data Cleaning**: Removed missing values, verified data types.
2. **Exploratory Data Analysis**:
   - Visualized distribution of life expectancy per country.
   - Heatmap of correlations between numeric features.
3. **Feature Selection**: Removed non-numeric and redundant columns.
4. **Model Building**: Trained a Random Forest Regressor.
5. **Evaluation**: Used MAE (Mean Absolute Error) for performance.

---

## 📈 Sample Code Snippets

```python
# Load data
import pandas as pd

df = pd.read_csv("data/life_expectancy.csv")

# Clean and prepare
df = df.dropna()

# Train/test split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X = df.drop(["Life expectancy ", "Country"], axis=1).select_dtypes(include='number')
y = df["Life expectancy "]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
```

---

## 📌 Conclusion

This project provides insights into how life expectancy is influenced by a combination of healthcare, economic, and lifestyle factors. A machine learning model helps quantify these relationships and offers a way to predict life expectancy based on measurable indicators.

---

## 📬 Contact

Feel free to reach out with suggestions or questions! **Project by**: Almeera

---

*This project is for educational purposes only. Dataset credit goes to the World Health Organization and Kaggle.*

