# -Class-DataProgramming
Project 2

Part I Motivation and Problem

Importance of the Problem
The problem we are solving is important to businesses that carry and sell avocados. To continue to make a profit on avocado sales, these businesses will benefit from a full understanding of market and price trends. 
In this way, businesses can adjust their avocado inventory, prices, and marketing to capture the largest profit. A strategy like this is made possible through our group’s analysis.

Why Pay to Solve the Problem
As a market analytics group, we can provide valuable market and price analysis, which includes an overview of a given market and marketing and sales strategies for the upcoming year. 
The overall avocado industry’s sales are growing. The potential benefit to businesses in avocado sales in terms of profit outweighs the cost of hiring our group to complete a market & sales strategy. 
Our group’s specific deliverables are outlined in the forthcoming solution section.

Problem Statement 
Our market analytics group is tasked with conducting a market and sales analysis of the avocado industry over time to offer solutions and strategies to avocado sellers leading to increased profits for those sellers. 

Part II Solution

High-level Description of Solution
Our solution to the problem stated above consists of a group of deliverables. Each of these deliverables is present with the goal to assist avocado sellers to stay in alignment when the industry and consumer trends in the upcoming years.
Specific deliverables:
Trend forecasting by type of avocado (conventional, organic)
Sales of Packaging analysis (bags: small, large)
Price forecasting 
Seasonal price & sales analysis
Visual representation for easy data comprehension 
Maintenance for future trends 

The Right Solution
Our solution is beneficial because it provides avocado sellers with a competitive advantage in the industry by considering all the insights gained and variables that influence their market and sales through analysis. We use past data of avocado sales to inform decisions around future strategy. 

Part III How it Works

Our Approach
The architecture of the approach is as follows:
Overview of data
Form deliverables based on available data
Pre-processing of data
Brainstorm machine learning (ML) models
Evaluate models against data, compare and select the most useful models
Finalize deliverables
Execute ML models 
Maintenance strategy
Provide final report 

Details of Approach and Inner Workings
We gather the data for the avocado industry and review the information collected to see what types of variables we have to work with. After the data overview, we shortlist the important aspects of the dataset that will best serve as a solution to the problem statement for the avocado sellers. We are then able to select relevant features and columns to gain insights and forecasts. Please refer to section 1 in the jupyter notebook. 
The next step in our approach is to pre-process and clean the data to make it easier to work with. Once the pre-processing is complete, we conduct a graphical and statistical analysis of the data to better understand it and gauge the avocado market. This includes looking at skewness, bar plots, box plots, etc. (Refer to Sections 2-4 in the jupyter notebook)
With the analysis of the dataset complete, we decide on the methods for price and sales forecasting because we have a time series dataset. From here, we build a crude Linear Regression model to test the approach on the data before fine-tuning the approach given a favorable result. Simultaneously, we research other models to employ and decide on two other regression models and ARIMA and FB prophet. 
The steps of FB Prophet: 
Pre-processing and cleaning of the data.
Choosing trends to predict: Conventional Avocados Price Prediction, Organic Avocados Price Prediction, Sales Prediction for Small/Medium, Large and Extra-Large Hass Avocados, Sales Prediction for Total Number of Avocados, all based on ‘Total US’ portion of the data.
FB Prophet requires the column names to be exactly: ‘ds’ for the DateTime column and ‘y’ for the other column of interest, so completed this for each of the six instances described above and fit and stored the predictions into ‘forecast’.
Visualization of the forecast using Plotly to generate an interactive graph with the corresponding Confidence Interval (CI).
Visualization of the nonlinear trends affecting the model using Plotly, which gave the year-on-year and seasonal trends. Please refer to section 13 in the jupyter notebook.
Steps for ARIMA: 
Looking at the plot of average avocado prices in the US, we form the assumption that past prices can be used to predict future prices with an AutoRegressive Integrated Moving Average (ARIMA) model. 
Using average price data for ‘Total US’ and ‘conventional’ type, an ACF (Autoregression correlation function) on the correlation between months returned a correlation of the first 4 months in a year. Plugging the values into the ARIMA model, a 95% confidence interval indicates a possible average price trend development for conventional type avocados. Please refer to the top part of section 12.1 in the jupyter notebook to see the ACF and PACF output.
Caveat: The CI range is very large starting at the beginning of the forecasting period. A possible reason for the poor performance is that the ARIMA model is unable to smooth the time series data. Please refer to the graphs at the bottom of section 12.1 of the jupyter notebook to see the 95% CI.

Justifications for Techniques
The data required us to do time series forecasting and we decided on the techniques based on that requirement.
ARIMA: An ARIMA model implementation was pursued due to the presumed correlation between past and future average prices after looking at the average US avocado prices chart. The intention was to smoothen the time series data and rid it of any seasonality/cyclicity. (Refer to Section in notebook)
FB Prophet: FB Prophet, based on the Bayesian based curve fitting method, was chosen as an excellent model for forecasting time series data on an additive model with nonlinear trends (seasonality). Since our data is seasonal and represented well by time series analysis, this model is a great match. Furthermore, this model handles outliers well, as seen from the graphs in (Refer to Sub-sections 13.6 - 13.8 in jupyter notebook) sales for the 3 types of Hass Avocado. Other overall benefits of this model include being open-source, fast, efficient, but those were not particularly taken advantage of in this problem.

Steps to Improve the Model 
After initial modeling, we further process the data and tune the models to help improve accuracy and effectiveness and leads to better visualization. These steps include:
Manipulate data by separating conventional and organic avocado data frames.
Drop any unnecessary categorical columns or convert them to dummy variables.
Scale the data using Standard Scaler fit, which removes the mean and scaled to unit variance. 

Part IV Evaluation and Discussion

Results of Models Based on Evaluation Metrics 
Our models and analysis are based on forecasting. We are not using an evaluation matrix because we aren’t classifying that data, instead, we use error values to evaluate the model results. 
Three error values used include: mean absolute error, mean square error, and root mean square error.
We then compare different models and graphically interpret them.

Model Performance
Models that Performed Well 
Regression models (Refer to Section 11 in notebook): Random forest regressor, simple linear regression, and XGBoost regressor perform above average or better. 
Generally, Random Forest works well on large datasets, works well with missing data by creating estimates for them, and produces better results. Random forest is an ensemble of decision trees meaning many trees constructed in a certain “random” way form a Random Forest. Each tree is created from a different sample of rows and at each node, a different sample of features is selected for splitting. Each of the trees makes its own individual prediction. These predictions are then averaged to produce a single result. 
FB Prophet: FB Prophet lived up to its promise, and delivered time-series graphs representing the trends very well. The secondary graphs showing the two components: 1) Year on Year trend, 2) Seasonality, were a valuable addition for analysis/deciding on the sales and marketing strategies.

Situations Where Models Did Not Perform Well
ARIMA model does not perform as intended here because the assumption that past price values can predict future values is unsupported, or because we are unable to successfully smoothen the time series data and remove any seasonality that may negatively affect this model.
FB Prophet leaves us unable to evaluate its prediction numerically (like rmse) because most of the operations are done under the hood.

Key Takeaways 
With our analysis we have found the following takeaways that can add value to the marketing and sales strategies of avocado sellers:
Avocado sales and prices follow a seasonal pattern, peaking in November and falling considerably by February. 
Sales of the extra-large avocados are falling. 
Sales of small bags of avocados are higher than large bags. Although, our data lacks the description of the number of avocados per small and large bag so it is uncertain which should be eliminated or made more available. 
Forecast prices with good accuracy and have a reliable model for the future.








References 

Kaggle code: 
https://www.kaggle.com/samuelbelko/predicting-prices-of-avocados
https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7

Standard Scaler fit: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html



