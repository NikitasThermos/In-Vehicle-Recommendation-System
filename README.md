# Coupon Recommendation System
Machine Learning project for 'Data Mining' university course
<h3>Tools</h3>

[![Tools](https://skillicons.dev/icons?i=py,sklearn,tensorflow)](https://skillicons.dev) 

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Processing](#processing)
4. [Results](#results)
5. [Deployment](#deployment)
6. [Dataset Features](#features)

<a name="introduction"></a>
## Introduction
The project focuses on comparing different simple Machine Learning models on a small dataset with the goal of identifying how much perfomance we can get with simple solutions. Initially we perform a fast preprocessing of the data to transfom them into a correct format needed by ML models and gain usefull information to improve the perfomance of the models. Then we use models from Scikit-Learn as well as a Dense Network created with TensorFlow. 

<a name="Dataset"></a>
## Dataset

<a name="processing"></a>
## Data Processing

<a name="results"></a>
## Results

<a name="deployment"></a>
## Deployment

<a name="features"></a>
## Dataset Features
| Feature | Description | 
| :--- | :--- |
| destination | Driver's destination when the coupon was offered
| passenger | Other passengers in the vehicle
| weather | Type of weather when the coupon was offered
| temperature | Temperature when the coupon was offered
| time | The time of day when the coupon was offered 
| toupon | The type of coupon that was offered 
| expiratioin | How long the coupon is valid for
| gender | The gender of the driver
| age | The age of the driver
| maritalStatus | Marital Status of the driver
| has_children | If driver has childer
| education | Education status of the driver
| occupation | Driver's occupation
| income | Driver's annual income
| car | Type of vehicle
| Bar | Number of times drives has been to a bar per month
| CoffeeHouse | Number of times drives has been to a coffee house per month
| CarryAway | Number of times driver got take-away food per month
| RestaurantLessThan20 | How many times driver has been to a restaurant with expense less than 20$ per person
| Restaurant20To50 | How many times driver has been to a restaurant with average expense per person between 20$ and 50$
| toCoupon_GE5min | If the driving distance to the coupon is greater than 5 minutes
| toCoupon_GEQ15min | If the driving distance to the coupon is greater than 15 minutes
| toCoupon_GEQ25min | If the driving distance to the coupon is greater than 25 minutes
| direction_same | Whether the direction to coupon is the same as the current direction
| direction_op | Whether the direction to coupon is opposite to the current direction
| Y | Whether the coupon is accepted
