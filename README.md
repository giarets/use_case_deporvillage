# Use case deporvillage

__Project structure__  

```
├── utils
│   ├── utils_features_family.py  
│   ├── utils_features.py
│   ├── utils_models.py
│   ├── utils_plots.py
│   ├── utils_preprocessing.py
│
├── 1_data_exploration.ipynb
├── 2_model_naive.ipynb
├── 3_model_lightGBM.ipynb
├── 2_model_hierarchical.ipynb
├── 2_model_hierarchical_categories.ipynb
```



> __Project description__:  
> You are required to generate a prediction of the total amount to sell during the year
2024-09-01 → 2025-08-31 for each brand+family. The goal of this prediction is to generate a
budget (per brand+family) for the purchasing team.  
> 
> You are given two data files, one with information about products, and another containing
historic sales data.  
> 
> Please hand in one (or several, if required) ipynb file with the data exploration, modelling
and outcome analysis; and a data file with the generated predictions.  
> 
> The developed ML solution does not need to be finetuned, since the goal is to evaluate your
approach to the process, not the specific result.  
> 
> Then answer the following questions:  
> 1. What metric have you used and why have you selected this particular metric? Is
there any drawback that comes from using this metric?  
> 2. What error can we expect from the generated predictions? How many units do you
estimate will be left unsold at the end of the year? How can we minimize this?  
> 3. If you had the time to develop this project further, what improvements would you
consider testing?  
> 4. How would you make these predictions available to the purchasing team? Please
write a few pros and cons of every alternative proposed.  
> 5. How would you evaluate the performance of this model in production? How would
you justify those numbers to the purchasing team?  
> 6. If there is any pandemic or economical crisis, which strategy would you propose to
make sure the model can be adapted to it?   


## Introduction

Multiple approaches would be possible. In this case, since we have to make predictions a 
year ahead, we would like to formulate it in forecasting terms: the time component will
be taken into account without shuffling dates. Features will have lagged values and we will
be careful with data leakage issues.  

Specifically this would be a hierarchical forecasting project. Hierarchical because of the
structure of the data which could be grouped following multiple dimensions or, to say it
differently, represented with a DAG (directed acyclic graph).  
Items can be aggregated at the brand + family level, then into brands and then globally. 
Or into families and the globally. Same with other categorical columns.

The forecast level of aggregation is *Brand + Family*. Even if this already aggregating 
multiple products, it still generates many intermittent time series with low volumes and
high variance. Besides, since the data is daily, predicting a year ahead means predicting 365 days ahead.
Which is a lot.  
Therefore it would be better to aggregate the time dimension to weekly or monthly data.  
Both the weekly and monthly approaches were tested and the code is general enough to manage them.  
The weekly approach was later discarded.  

In principle we have 2 possible approaches:

#### OPTION 1: Predict Quantity at Brand + Family Level, then Convert to Revenue  

1.	Aggregate historical quantity sold at (brand, family, date).
2.	Predict total quantity for the future period. Target Variable: total_quantity per brand/family for the forecast period.
3.	Convert to revenue using an estimated PVP per (brand, family), such as a Weighted PVP based on sales distribution:
$$\text{AvgPvp} = \frac{\sum (\text{quantity} \times \text{pvp})}{\sum \text{quantity}}$$

- Pros:  
	•	More granular, allowing adjustments based on price changes.  
	•	If PVP changes, we can easily recalculate revenue.  

- Cons:  
	•	Requires an extra step and assumes PVP remains constant or follows a simple trend.    

This is probably better is PVP fluctuates.  


#### OPTION 2: Directly Predict Revenue (Sales)  

Target Variable: $\text{total revenue} = \text{total quantity sold} × PVP$  

1.	Aggregate historical revenue at (brand, family, date).
2.	Predict total revenue for the future period. 

- Pros:  
	•	Simpler approach with fewer calculations.
	•	Directly aligns with the budgeting goal.  
    •   No need to worry about varying PVP within a brand + family.

- Cons:  
	•	If prices fluctuate, the model won’t capture it properly.  
	•	Less flexibility—harder to analyze quantity vs. price effects separately.  

Assumes PVP is stable.   

Since the price of all items is constant over time, we will directly predict the revenues.  



## Problem formulation

__Target variable__: *pvp*  
__Frequency__: *Monthly*  
__Aggregation key__: *Brand + Family*  
__Forecasting horizon__: *12 months*  


## Modelling
### Naive models

Baseline before implementing more advanced models.  
Note that in many forecasting problems these baselines might be not so easy to beat.  

- NaiveLag: propagates the last observation
- NaiveRollingMean: propagares the average of the last observations

### LigthGBM

Classical boosted approach.  
It adapts well to many situations and might be good here that we have many categorical columns.  
It might also be a good idea consideriong that the forecasting horizon is very large.  
Unfortunately there is not a lot of data if we aggregate at the monthly level.

### Hierarchical models

What It Is   
- Hierarchical forecasting aggregates data at higher levels (e.g., brand level or total sales) to compensate for sparsity at lower levels (e.g., brand + family).   
- Predictions are made at the higher levels and disaggregated down to smaller groups (e.g., brand-family) using proportions.   
  
Steps to Implement:

1.	Aggregate to Higher Levels:   
	- Aggregate sparse brand-family data up to broader levels:   
	- Brand level: brand_total_sales  
	- Family level: family_total_sales  
	- Overall total: global_total_sales  
	- Use the aggregated time series to train the model.  
2.	Forecast at Higher Levels:   
	- Predict sales at the aggregated levels using richer historical data.  
3.	Disaggregate Forecasts:  
	- Split the higher-level forecasts back down to the brand-family level using historical proportions. For example:
$\text{brand\_family\_forecast} = \text{brand\_forecast} \times \frac{\text{brand\_family\_sales}}{\text{brand\_total\_sales}}$  
4.	Smooth Proportions for Stability:  
	- Use smoothed historical proportions (e.g., rolling averages) to prevent overfitting to noisy historical proportions.   


Multiple implementation were tested:
- Forecast with the overall time series vs forecast with the brand time series
- Forecast with Sarimax vs forecast with Exponential smoothing

## Conclusions

None of the models was fine tuned and we are just analysis gross results.  
However it looks like the hierarchical model gives the better results.  
For a complete storytelling, please refer to the notebooks.


# Questions

> 1. What metric have you used and why have you selected this particular metric? Is
there any drawback that comes from using this metric?  

baila baila

> 2. What error can we expect from the generated predictions? How many units do you
estimate will be left unsold at the end of the year? How can we minimize this?  


> 3. If you had the time to develop this project further, what improvements would you
consider testing?  


> 4. How would you make these predictions available to the purchasing team? Please
write a few pros and cons of every alternative proposed.  


> 5. How would you evaluate the performance of this model in production? How would
you justify those numbers to the purchasing team?  


> 6. If there is any pandemic or economical crisis, which strategy would you propose to
make sure the model can be adapted to it?  