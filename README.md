# Predict Honey Price - Machine Learning Subject- FIEK 2023/2024

## Team Members
- Eriona Osaj
- Edon Budakova
  
## Overview
This Jupyter Notebook is part of a project aimed at predicting the price of honey using advanced algorithms like CatBoost and LightGBM. The dataset is sourced from Kaggle and includes various features related to honey characteristics and its purity, which are crucial in predicting the price.

## Objective
The goal of this project is to apply various data preprocessing techniques to prepare the honey dataset for predictive modeling. The notebook will cover preprocessing and exploratory data analysis, prediction models and evaluation.

## Dataset Description
The dataset includes 247903 rows with the following columns with their respective data ranges:
- **CS (Color Score):** Ranges from 1.0 to 10.0, indicating the color of the honey.
- **Density:** Ranges from 1.21 to 1.86 grams per cubic centimeter at 25°C.
- **WC (Water Content):** Ranges from 12.0% to 25.0%.
- **pH:** Ranges from 2.50 to 7.50.
- **EC (Electrical Conductivity):** Measured in milliSiemens per centimeter.
- **F (Fructose Level):** Ranges from 20 to 50.
- **G (Glucose Level):** Ranges from 20 to 45.
- **Pollen_analysis:** Indicates the floral source of honey, including Clover, Wildflower, etc.
- **Viscosity:** Ranges from 1500 to 10000 centipoise, with 2500-9500 being optimal for purity.
- **Purity:** The target variable, ranges from 0.01 to 1.00.
- **Price:** The price of honey, which is the primary variable we aim to predict.


## Dataset Attributes

**Categorical Attributes:**

_Nominal:_
- Pollen_analysis

_Ordinal:_


**Numerical Attributes:**

_Interval:_
- EC
- pH

_Ratio:_
- Price
- Purity
- Viscosity
- G
- F
- WC
- Density

  ## Data Integration
In our project, data integration plays a pivotal role in ensuring that we have a comprehensive and unified dataset for analysis. 

- **Single Source Integration**: Our dataset is sourced exclusively from Kaggle, which simplifies the integration process. The dataset encompasses extensive information about vehicle fuel consumption ratings and emissions, making it a valuable resource for our analysis. We imported the dataset into our analytical environment using the following command:
   ```python
    dataset = pd.read_csv('Data/honey_purity_dataset.csv') 

## Data quality

In our effort to ensure the highest data quality for our analysis, we conducted an extensive examination of the dataset. Below are our key findings and the methodologies applied in each aspect of data quality assessment.

### Missing Values

We didn't detect missing values in our dataset.
<img width="170" alt="image" src="https://github.com/ErionaOsaj/Private.FIEK.ML.HoneyPricePrediction/assets/27639068/5bc549ea-6242-4d9d-9b78-08154a9f3409">

### Duplicate Records
- **Duplicate Check**: We verified that there are no duplicate records in our dataset using the `dataset.duplicated().sum()` function, ensuring the uniqueness and integrity of our data.

<img width="147" alt="image" src="https://github.com/ErionaOsaj/Private.FIEK.ML.HoneyPricePrediction/assets/27639068/4e7a0de8-707d-4b32-afe3-7c7640913ecd">

