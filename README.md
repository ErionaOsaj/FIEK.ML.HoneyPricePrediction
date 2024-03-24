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


## Aggregation

In our honey dataset, we have applied data aggregation techniques to effectively summarize and understand the data. This process helps us to draw insights into the various factors that may influence the price of honey based on its pollen source.

- **Price Aggregation by Pollen Analysis**: We have aggregated the price data based on 'Pollen_analysis', which refers to the floral source of the honey. By doing so, we can discern the average, minimum, and maximum prices associated with each type of honey. This aggregation is crucial as it reveals potential pricing trends related to the honey's origin.

The aggregated data provides valuable insights, such as:
- Understanding the pricing structure across different types of honey.
- Identifying which types of honey are more expensive on average and which are more affordable.
- Assessing the range of prices within each honey type to understand market variability.

These aggregated statistics are instrumental for stakeholders to make informed decisions, such as which honey types to prioritize for marketing or further quality assessments. It also enables producers to benchmark their products in the market context.

Through this summarized view, we aim to facilitate a comprehensive understanding of the factors influencing honey prices, enhancing our analytical capabilities for predictive modeling and trend analysis.

We also performed aggregation by Pollen Analysis in all other numerical columns.

## Outlier Detection and Data Exploration

### Outlier and skewness etection using IQR (Boxplots)  

Outliers can significantly impact our analysis, leading to skewed results. Detecting and handling these is crucial for maintaining the integrity of our findings.

We used boxplots to visualize and detect outliers in our numerical data. Here's how we did it:

- `plt.boxplot(dataset['Engine Size (L)'], showmeans=True)`: This command generates a boxplot for the 'Engine Size (L)' column and shows the mean value. The boxplot provides a visual summary of the central tendency and dispersion of the data, as well as potential outliers.
  
  The following values were extracted from the 'Engine Size (L)' boxplot:
  - Median: The middle value of the dataset when it is ordered from least to most.
  - Mean: The average of the dataset, indicating central tendency.
  - Minimums: The lowest value within the range that is not considered an outlier.
  - Maximums: The highest value within the range that is not considered an outlier.

- `extract_boxplot_values(bp)`: This custom function extracts and prints the median, mean, minimum, and maximum values from the boxplot object. It uses the matplotlib `boxplot` object to access the statistical properties of the plotted data.

Here is an example of boxplot:
<img width="236" alt="image" src="https://github.com/ErionaOsaj/Private.FIEK.ML.HoneyPricePrediction/assets/27639068/1f1a98a1-ebd0-4d4c-8fb1-0b34582f7e1e">

With this method we didn't detect outliers data and no significant skewness. 

## Tools and Environment
- Jupyter Notebook for analysis.
- Python v3.x for scripting.
- Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and similar libraries for data manipulation and visualization.

Before running the analysis, ensure that you have the necessary Python libraries installed. Below are the commands to install each library using `pip`, which is the Python package installer.

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install tabulate
pip install scipy
pip install scikit-learn
```



## Additional Resources
- [Kaggle Dataset: Predict Purity and Price of Honey](https://www.kaggle.com/datasets/stealthtechnologies/predict-purity-and-price-of-honey/data)

## References
This project uses data available on Kaggle, and it is assumed that the dataset is used in accordance with Kaggle's terms of service.

For any questions or additional information, please contact the project contributors.


