# HKBU-FIN7790-GroupE

## Project Describe
We collect the dataset containing real financial indicators of 2030 publicly traded companies over 11 consecutive quarters.
Including budget forecasting, financial health, loan default prediction, etc.


Our target is to predict the companies’ financial data points for the last (closest to today) reported quarter. Each quarter in the dataset is represented by 9 financial data points such as TOTAL_ASSETS, TOTAL_LIABILITIES, EBITDA, etc. and the quarter indicator is the prefix (Q0, Q1, … Q10) where the “0” indicates the most recent quarter in the dataset.

### Reference

```
@misc{financial-performance-prediction,
author = {Danil Zherebtsov},
title = {Financial Performance Prediction},
publisher = {Kaggle},
year = {2024},
url = {https://kaggle.com/competitions/financial-performance-prediction}
}
```

## Data Describe
### Basic information
The <keybord>train.csv</keyborad> contain 212 columns and 1624 rows, total have:
```
float64: 207 | object: 4 | int64: 1
```
Following are the industry that each sector have:
![SectoreCategory.png](resourcese%2FSectoreCategory.png)

### Preprocess
We explore the data and we found that this data contain lots of empty value.
Only 431 rows contain full of data. It is quite hard to impute the data by using traditional way, 
because these data are related to real world and if we impute with mean/max/min could manually cause outliers.

For example: the totalRevenue of missing data is the largest, if we impute with common way, may cause the 
![OutlierExample.png](resourcese%2FOutlierExample.png)
So

