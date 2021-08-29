# movie-revenue-prediction

The data set is composed of movies titles and features like budget, country, language, mpaa rating etc. THe dependent column "Total Revenue" is the sum of columns "US revenue" and "Internation revenue".

## Data preparation
The dataset is cleaned, categorical variables encoded. 
Split into training and testing data
Assumptions of linear regression are tested.

## Modelling
Linear regression and Xgboost models are trained on the data and evaluated for performance. 

## Results
Linear regression performs slightly better in terms of the rmse. 
Feature importance is studied for predicting future movie revenue.
