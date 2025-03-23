# National Pollutant Release Inventory (NPRI) Classification Models for Criteria Air Contaminants
## Project Overview
This project aims to predict trends in the release of criteria air contaminants (CACs) in Canada using data from the National Pollutant Release Inventory (NPRI). These contaminants include sulfur dioxide (SO₂), nitrogen oxides (NOₓ), volatile organic compounds (VOCs), particulate matter (PM), and carbon monoxide (CO). The goal is to predict whether emissions of these pollutants will increase, decrease, or remain stable in 2023, which is crucial for regulatory and environmental decision-making.
-> We chose to focus on these specific contaminants because they are commonly known as "criteria air contaminants" that are used to assess air quality and public health, making them a priority for environmental regulations.

## Dataset
The dataset used in this project is sourced from the National Pollutant Release Inventory (NPRI), which tracks pollutants released to air, water, and land in Canada. It contains emissions data across multiple industries and regions, as well as other environmental variables.

## Key Dataset Features:
•Year: Year of data reporting.
•Region: Geographic location (e.g., province, city).
•Industry Classification: Type of industry (e.g., manufacturing, agriculture).
•Pollutants: Emission amounts of pollutants (SO₂, NOₓ, VOCs, PM, CO).
•Emissions: Amount of pollutants released (in metric tons).
•Emission Change: Difference in emissions between 2022 and 2023.
## Target Variables: 
The model focuses on predicting the trends for the following CACs:

Sulphur Dioxide (SO₂)
Nitrogen Oxides (NOₓ)
Volatile Organic Compounds (VOCs)
Particulate Matter (PM)
Carbon Monoxide (CO)

## Features:
Key features include geographic location, industry classification, reported amounts of emissions, and year of data reporting, among other environmental and industry-specific factors.
-> We chose the most relevant columns for this project, as emissions data and industry characteristics are the primary factors influencing pollutant release. Including too many columns could introduce noise and complexity.

## Project Structure
### Data Preprocessing: 
Initial steps include data cleaning, handling missing values, and normalizing data where necessary.
1. **Loading Data:**
   Load the dataset into a Pandas DataFrame.

   ```python
   import pandas as pd
   data = pd.read_csv('npri_data.csv')
   print(data.head())
   ```
2. **Handling missing values:**
   Handle missing values using imputation or dropping rows.
   ```python
   data.fillna(data.mean(), inplace=True)
   ```
3. **Feature engineering:**
   ```python
   # Select relevant columns
   releases_filtered = releases_df[['Reporting_Year / Année', 'Substance Name (English) / Nom de substance (Anglais)',
                                 'Number of employees',
                                 'Air_Releases', 'Land_Releases', 'Water_Releases',
                                 'Total_Releases'
                                      ]]
   # Melt the DataFrame to long format
   releases_long = releases_filtered.melt(
   id_vars=['Reporting_Year / Année', 'Substance Name (English) / Nom de substance (Anglais)',
             'Number of employees'],
   var_name='Release_Type',
    value_name='Release'
   )
   # Convert the year to datetime format
   releases_long['Reporting_Year / Année'] = pd.to_datetime(releases_long['Reporting_Year / Année'], format='%Y')

   # Inspect the transformed DataFrame
   print(releases_long.head())
   # Convert the year to datetime format
   releases_long['Reporting_Year / Année'] = pd.to_datetime(releases_long['Reporting_Year / Année'], format='%Y')
   # Inspect the transformed DataFrame
   print(releases_long.head())
   ```
### Exploratory Data Analysis (EDA): 
Visualizations and summary statistics to understand the distribution of contaminants and key features influencing emissions.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing emission distribution by year
sns.boxplot(x='Year', y='Emissions', data=data)
plt.title("Emission Distribution by Year")
plt.show()
```

### Modeling: 
Implemented classification models to predict whether emissions for each contaminant are likely to increase, decrease, or remain stable in 2023.
```python
def process_substance(substance_name, substance_data):
    # Step 1: Extract data for the given substance
    data = substance_data[substance_name]

    # Step 2: Prepare the features and target variable
    X = data[['Lag_1', 'Lag_2', 'Lag_3', 'Number of employees']]

    # Target variable
    y = data['Release_Category']

    # Combine X and y for shuffling
    combined_data = pd.concat([X, y], axis=1)

    # Get the best parameters and estimator
    best_model = random_search.best_estimator_

    # Make predictions with the best model for validation phase
    y_val_pred_best = best_model.predict(X_val)

```
### Feature importance:
We can evaluate the feature importance in our random forest model:
```python
# Get feature importance
importances = model.feature_importances_

# Visualize the importance
plt.bar(X.columns, importances)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```

### Model Evaluation: 
Accuracy, F1 score, and confusion matrix are used to assess model performance.
```python
# Evaluate the model on the validation set
    print(f"Validation Classification Report for {substance_name}:")
    print(classification_report(y_val, y_val_pred_best))

    print(f"Validation Confusion Matrix for {substance_name}:")
    print(confusion_matrix(y_val, y_val_pred_best))

    # Make predictions on the test set to evaluate the final model
    y_test_pred_best = best_model.predict(X_test)

    # Evaluate the model on the test set
    print(f"Test Classification Report for {substance_name}:")
    print(classification_report(y_test, y_test_pred_best))

    print(f"Test Confusion Matrix for {substance_name}:")
    print(confusion_matrix(y_test, y_test_pred_best))
```

## Classification Models
The following machine learning algorithms were explored for classification:

Decision Tree Classifier
Random Forest Classifier
Logistic Regression
Support Vector Machine (SVM)

## Results
Achieved the highest accuracy with *Random Forest Classifier*.

## Dependencies
-> Python 3.8+

-> Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Results and Visualizations
Visualizations were created to represent the emission trends of CACs by region and industry, showing insights into high-pollution areas and industry-specific emissions.

## Future Work
Possible improvements could include:

•Fine-tuning models using cross-validation.

•Experimenting with ensemble methods to improve accuracy.

•Expanding features to include socio-economic data for deeper analysis.

## Acknowledgments
This project was conducted as part of the Machine Learning Analyst diploma program at NorQuest College, with a focus on environmental data analysis and prediction.
