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

### Boxplot Visualization of numerical columns  by Pollen Analysis

We performed boxplot vizualization of numerical columns by Pollen Analysis. We will describe one of examples and benefits from this vizualization.

#### Overview

The boxplot visualization presents a comparison of honey density across different pollen sources. By displaying the data distribution for each type of pollen, we can discern variability and central tendency, which are critical for understanding how the type of pollen correlates with honey density.

#### Insights from the Boxplot Visualization of Density by Pollen Analysis


The boxplot for each pollen type (such as Blueberry, Alfalfa, Chestnut, etc.) shows the median, quartiles, and potential outliers for the density of honey. The following observations and benefits can be derived from this visualization:

- **Comparative Analysis**: We can compare the central tendency (median) of honey density across various pollen sources. This comparison may reveal which pollen types tend to produce denser honey.
- **Density Ranges**: The range of density values, represented by the 'whiskers' of the boxplots, indicates the variability in density measurements for each type of pollen.
- **Outliers Detection**: Any potential outliers in the honey density data are identifiable as points outside the 'whiskers'. These can indicate data points that are exceptionally different from the general distribution and may warrant further investigation.
- **Data Dispersion**: The spread of the boxplot (interquartile range) shows how concentrated or dispersed the density values are around the median. A narrow box indicates low variability, whereas a wide box indicates high variability.
- **Purity Implications**: As density can be a factor in determining the purity and quality of honey, analyzing the distribution of this metric across pollen sources can provide insights into the quality control and standardization of honey products.
![image](https://github.com/ErionaOsaj/Private.FIEK.ML.HoneyPricePrediction/assets/27639068/36eca3b3-4dc0-482c-9429-ac8bd4543ffa)

........
#### Conclusions from the Figure

- **Uniformity**: The boxplots show a relatively uniform distribution of density across different types of honey, with no significant deviation in median values, which suggests that pollen type might not drastically affect the density of honey.
- **Consistency**: There appears to be consistency in the honey density range across the different pollen types, suggesting standard quality and processing methods.
- **Quality Control**: The absence of extreme outliers in most pollen categories implies good quality control in honey production. It can indicate that the honey's density is being maintained within expected parameters regardless of its floral source.

This boxplot serves as a powerful exploratory tool, providing a visual summary that aids in understanding the complex relationships between honey density and its pollen source, which is pivotal in further analysis and predictive modeling endeavors.
### Boxplot Visualization of Honey Prices by Pollen Analysis

#### Overview

We have created a boxplot visualization to analyze the distribution of honey prices for each type of pollen source. This graphical representation is instrumental in understanding how the pollen type might influence the market price of honey.

#### Insights from the Boxplot

The boxplot provides a visual summary of the prices for each pollen category, revealing the central tendency, spread, and potential outliers within each group. From the visualization, we can derive several key insights:

- **Price Variation by Pollen Type**: There is noticeable variation in median prices among the different types of pollen sources. Certain categories, such as Manuka and Heather, show higher median prices, which might suggest a premium market value for these honey types.
- **Consistency Across Categories**: The interquartile ranges, indicating the middle 50% of the price data, are fairly consistent for some pollen categories, while others show greater variability.
- **Outliers Detection**: The boxplots indicate the presence of outliers in several pollen categories. These outliers represent honey prices that are significantly different from the general pricing trend within that category and could be due to unique quality factors or market demand.
- **Affordable and Premium Options**: Categories like Blueberry and Alfalfa appear to have lower median prices, potentially indicating more affordable options in the honey market. In contrast, categories with higher median prices and upper quartiles, such as Manuka, may represent premium product offerings.

 ![image](https://github.com/ErionaOsaj/Private.FIEK.ML.HoneyPricePrediction/assets/27639068/00532b5f-8310-45eb-bbe6-bfeeba5740e8)


#### Conclusions from the Figure

The boxplot visualization serves as a pivotal analytical tool for exploring the price dynamics in the honey market. It highlights that:

- Pollen source is a significant factor that could affect the pricing of honey.
- The presence of outliers may indicate special cases or premium quality honey that commands higher prices.
- Understanding these pricing patterns is crucial for producers, distributors, and retailers to position their products effectively in the market.

By analyzing these patterns, stakeholders can make informed decisions about which honey types to focus on for different market segments, ensuring alignment with consumer demand and market trends.

### Skewness

Skewness is a statistical metric that gives us an idea of the symmetry, or lack thereof, in the data distribution. The skewness coefficient is a single numerical value that reflects the shape of the distribution of values in a dataset.
With skewness coefficient we didn't notice any sign of skewness in our data.

Example: We calculate the skewness for the 'Cylinders' feature in our dataset as follows:
```python
skewness = dataset['Price'].skew()
print(f"Skewness coefficient: {skewness}")
```
The skewness value gives us an indication of the asymmetry level in the distribution:

- **Approximately Symmetric Distribution**:
  - A skewness value between -0.5 and 0.5 means the distribution is approximately symmetric.

- **Moderately Skewed Distribution**:
  - A skewness value between -1 and -0.5 indicates moderate negative skewness (left-skewed).
  - A skewness value between 0.5 and 1 indicates moderate positive skewness (right-skewed).

- **Highly Skewed Distribution**:
  - A skewness value less than -1 indicates a highly negative skewness (left-skewed).
  - A skewness value greater than 1 indicates a highly positive skewness (right-skewed).


 ## Data Sampling for Efficient Testing

To enhance the efficiency of our testing process, particularly when applying computationally intensive predictive models, we implemented a stratified sampling technique to create a smaller, more manageable subset of our dataset while maintaining its statistical properties.

### Sampling Strategy

Our stratified sampling approach ensures that each category of 'Pollen_analysis' is represented proportionally in the sampled dataset. Here's a brief overview of the sampling method used:

1. **Sample Size Determination**: We determined the size of our sample to be 10% of the original dataset. This size strikes a balance between representativeness and computational speed.

2. **Stratification**: To retain the original distribution of the 'Pollen_analysis' categories, we divided the data into stratified groups based on the unique pollen types.

3. **Sampling Per Category**: We then performed sampling within each stratified group, ensuring that each pollen category contributed an equal number of observations to the final sample, subject to the availability of data points in each category.

4. **Data Consistency**: After sampling, we reset the index of our new dataset to maintain data consistency and facilitate ease of analysis.

The resulting `dataset_sampled` maintains the diversity and distribution of the original data, allowing for accurate model testing and validation on a smaller scale.

### Benefits of Sampling

- **Speed**: By working with a smaller dataset, we significantly reduce computation time, which is particularly advantageous when iterating over different model configurations.
- **Scalability**: Sampling allows us to test our models on hardware with less computational power, thereby improving the scalability of our testing phase.
- **Model Validation**: It provides a way to quickly validate model performance before deploying models on the full dataset, ensuring efficient use of resources.

## Correlation Heatmap Analysis

### Overview

A correlation heatmap has been generated to visualize the pairwise correlations between various physicochemical properties of honey, its purity, and price. The heatmap uses shades of colors to represent the strength and direction of the correlations, providing an intuitive display of how these variables interrelate.

![image](https://github.com/ErionaOsaj/Private.FIEK.ML.HoneyPricePrediction/assets/27639068/d05ebf3e-911b-42b2-95a5-9b604349a1c6)


### Heatmap Interpretation

- **Diagonal Values**: The diagonal of the heatmap, highlighted in red, shows a perfect correlation of 1.00, as any variable is perfectly correlated with itself.
- **Correlation Coefficients**: Values on the heatmap range from -1 to 1, where:
  - **1** indicates a perfect positive correlation: as one variable increases, the other variable also increases.
  - **0** indicates no correlation: the variables do not have a linear relationship.
  - **-1** indicates a perfect negative correlation: as one variable increases, the other variable decreases.
- **Color Scheme**: The 'coolwarm' color scheme has been used, where warmer colors (reds) represent positive correlations, and cooler colors (blues) indicate negative correlations.
- **Notable Correlations**: 
  - The 'Purity' of honey shows a moderate positive correlation with 'Price', suggesting that purer honey tends to have a higher price.
  - The 'pH' level shows a weak negative correlation with 'EC' (Electrical Conductivity), indicating that as the pH level increases, the electrical conductivity tends to decrease slightly, albeit the relationship is not strong.
  - Most variables show little to no correlation with each other, which is represented by the color close to white.

### Conclusions from the Figure

The heatmap suggests that while some variables have a mild correlation with honey prices, most of the physicochemical properties and purity do not strongly influence the price in a linear fashion. This can imply that the factors determining honey prices are complex and may not be directly deduced from simple pairwise correlations. It also suggests that for predictive modeling purposes, non-linear relationships should be considered, and machine learning models such as LightGBM and CatBoost, which can capture complex patterns, would be suitable for such analysis.

The heatmap is a valuable exploratory tool that helps us to identify which variables may be worth investigating further for their impact on honey prices and purity, and aids in the preprocessing steps by identifying any multicollinearity between variables.

## Z-Score Distribution Analysis for 'CS' (Color Score)

### Overview

The provided plot illustrates the distribution of 'CS' (Color Score) within the dataset, analyzed through the lens of z-scores, which standardize the data points in terms of their distance from the mean, measured in units of standard deviation.

![image](https://github.com/ErionaOsaj/Private.FIEK.ML.HoneyPricePrediction/assets/27639068/2908edd9-f198-4f4d-a9f1-7ba34b9aa4da)

### Description of the Plot

- **Data Points**: Represented by blue dots, each point corresponds to an individual honey sample's color score within the dataset. The x-axis shows the index of the sample, and the y-axis indicates the standardized z-score of the 'CS' value.
- **Mean Line**: The dashed red line across the center of the plot denotes the mean of the 'CS' values. Since the z-score standardizes around the mean, this line represents a z-score of zero.
- **Standard Deviation Bands**: The shaded regions in the plot signify the area within one, two, and three standard deviations from the mean. These bands are colored in gradients from yellow to blue, with yellow representing the area within one standard deviation (68% of data points if normally distributed), green within two standard deviations (95%), and blue within three standard deviations (99.7%).
- **Outliers**: Outliers are typically defined as data points lying beyond the outermost band (more than three standard deviations from the mean). However, in this plot, there are no data points marked as outliers, indicating all 'CS' values fall within three standard deviations.

### Conclusions from the Figure

The plot demonstrates that the 'CS' values are tightly clustered around the mean, with a distribution that does not exhibit significant skewness or outliers. This implies a high level of consistency in the color score data, with no extreme values that deviate markedly from the norm.

With no outliers detected, the data for 'CS' does not show signs of extreme values that could potentially skew the analysis or predictive modeling efforts. This suggests that the 'CS' feature has a stable distribution, which might not require outlier treatment before proceeding with further data analysis or inclusion in machine learning models.

This z-score distribution plot is an essential diagnostic tool, confirming the data quality and distribution characteristics for the 'CS' feature, thereby validating it for use in predictive algorithms such as LightGBM and CatBoost without the need for additional data transformation.



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
