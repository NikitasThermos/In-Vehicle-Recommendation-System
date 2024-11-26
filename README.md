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
The dataset is from the paper " Wang, Tong, Cynthia Rudin, Finale Doshi-Velez, Yimin Liu, Erica Klampfl, and Perry MacNeille. 'A bayesian framework for learning rule sets for interpretable classification.' The Journal of Machine Learning Research 18, no. 1 (2017) " which is available [here](https://www.jmlr.org/papers/volume18/16-003/16-003.pdf) and it was collected via Amazon Mechanical Turk. The dataset containts attributes about drivers when offered a coupon for a particular venue and whether the accepted or declined the offer. Specifically, the dataset contains some user attributes (gender, age, income, occupation etc.) and some contextual attributes (weather, temperature, time, destination etc.). A full list of the attributes can be found on the [Dataset Features](#features) section. The dataset contains 12684 cases from which 7210 were accepted (marked as '1' in the Y attribute) and 5474 were declined (marked as '0' in the Y attribute). The goal is to identify whether the user will accept the coupon based on these attributes. Thus the problem can be classified as a binary classification task.         

<a name="processing"></a>
## Data Processing
In the first step of preprocessing we dropped two features. First the 'car' feature has 99.23% missing values so we thought that it cannot give great information and trying to fill the all these missing values could mislead the results. Secondly the 'direction_op' feature is the exact opposite of the 'direction_same' feature, so keeping them both will not give us any extra infromation for the models. After that we worked with the three attributes that describe the distance to the coupon (toCoupon_GE5min , toCoupon_GE15min, toCoupon_GE25min). When the distance to the coupon is greater than 25 minutes (toCoupon_GE25min is marked as 1) it is implied that it will also be greater than 5 and 15 minutes (both toCoupon_GE5min and  toCoupon_GE15min are also marked as 1). The same happens when the distance is greater than 15 minutes which implies that is also greater than 5 minutes. For that reason we decided to combine those attributes by adding them together, creating a new attribute 'distance_to_coupon' and dropped the original attributes.
Because the dataset contains a lot of categorical attributes we needed to encoded them before inserting them in the ML models. In some cases we decided to encode the attributes by perseving a meaningful order (e.g. temperature, time, age). 

Because we did not have access to a seperate test set, we created a subset from the original set by selecting randomly 20% of the instances. This was done before the preprocessing took place to avoid any data leakage. That set was only used for testing the final perfomance of the models. 

Finally, for the DNN model we also standardized the features because DNNs are more vulnurable to different attribute scales ( we also used ReLU activation in all hidden layers to avoid that problem). Also for the DNN we created a validation set (10% of the remaining training set) so that we can keep track of the model's perfomance during training and use Early Stopping when the model stops increasing perfomance.    

<a name="results"></a>
## Results
For the comparison of the models we used Precision and Recall which are the most popular metrics for binary classification problems as well as a combination of those to create the F1 and ROC-AUC scores.  

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
