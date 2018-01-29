# Machine Learning Engineer Nanodegree
## Capstone Proposal
Quang Vu

## Investment and Trading Capstone Project

### Domain Background
_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

Stock market price is both interesting and challenging to predict. Traditionally, it is analyzed with statistical model, then tested with series of back-testing, which is a process of applying the model with historical data to measure the performance of the model. This project aims to use a LSTM (Long Short Term Memory) model to automate that process, and measure how much information can be captured by the LSTM, comparing to statistical model.

The project's implementation will apply the method mention on this paper: [Financial Market Time Series Prediction with RNN](http://cs229.stanford.edu/proj2012/BernalFokPidaparthi-FinancialMarketTimeSeriesPredictionwithRecurrentNeural.pdf). And the statistical method used for benchmark is [Autoregressive Integrated Moving Averate (ARIMA) model](https://www.r-bloggers.com/forecasting-stock-returns-using-arima-model/).


### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

The stock market has some properties on its data for each day: Open price, Close price, Adjusted price. In this project, I will use a RNN with LSTM to process a window interval (7 days, 30 days, etc...), and try to predict the stock price for the next day. A naive prediction would be taking an average of all the previous price in the window as a prediction for the next day's price:

```
price[t] = avarage(price[t-1] + price[t-2] + price[t-3]...)
```

An improve of that method is a ARIMA model, which takes into consideration a coefficient for each day price:
```
price[t] = a1*price[t-1] + a2*price(t-2] + a3*price[t-3] ...
```

We will see if an LSTM model can make a better prediction than a naive guess and an ARIMA model. Also, we will see which properties or which hyperparameter can be optimzed to help the performance of the model.

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

The New York Stock Exchange (NYSE) in 2010-2016 will be used for this project. The dataset can be downloaded on Kaggle (https://www.kaggle.com/dgawlik/nyse/data). The data is chosen because of the record of Open/Close/Low/High price for each day on the market during 2010-2016. In addition, there is also extra information about the securities price, splited-price, and the information about the company's earning, expense, profit... that can be served as extra parameters for the model.

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

An LSTM model will look at a sequence of data in an interval (for example: 7 days), and try to capture the trend of the price and try to make the prediction for the next day. The accuracy of the prediction will be base on the actual price from the data for that day. The testing data will be split at the end of the actual data, to simulate predicting future price given historical prices. The training data will be taken from the data, with its validation data will be the price of next day.

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

The naive guess and the ARIMA model makes prediction base on previous price data, and compare them to the actual data obtain. Their metrics are obtain from how closely the predictions follow the actual price:

![Visual result](https://i2.wp.com/www.quantinsti.com/wp-content/uploads/2017/03/Actual-Vs-Forecasted-Returns.png)

![Data](https://i1.wp.com/www.quantinsti.com/wp-content/uploads/2017/03/Returns-Series.png)


### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

While the metric of the benchmark model is well-defined, there are different metrics and less measurable metrics for using the LSTM model. The most important performance for evaluating the LSTM model, would be whether the model correctly capture the trend of the price. For example, if the model is able to predict the downward or upward trend of the price. Secondly, how much difference between the model's prediction and the actual result. The direction of the difference should also be taken into account. Thirdly, how lag the prediction is, ex: how many days after the price change, the model would be able to reflect/predict those changes.

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

The project will be implemented follows these outline steps:

**Exploring the data**
- How large is the data?
- How many companies involve in the data?
- Is price on each company in the same range, or vastly different?
- How frequently is the data collected?
- Is there any anomily in the data?

**Prepare the data**
- Load the data
- Can the data be processed on my machine? Can the data be processed on my GPU? Do I need to sample the data to make it smaller?
- Does the data require normalize/denormalize?
- Prepare the training and testing data
- Does the data need to be converted to particular shape?

**Building the model**
- LSTM will be used, and mainly rely on Keras implementation for converting the LSTM model to Tensorflow backend.
- Determine how many LSTM layer is needed
- Determine if Dropout is needed for LSTM each layer, and what rate is best for the model.

**Train the model**
- How many epoch is needed? And how many epoch my machine can run. Will more epoch always better?
- If the training takes very long time, determine how to capture the training and resume if needed, without restart from scratch.
- Is there a way to visualize the on-training process?
- Is there a method to make model run faster?
- Is there a parameter to improve the above metrics of the model?

**Visualize the result**
- Make a naive prediction as a baseline benchmark.
- How is the result from the LSTM model compare to the benchmark?

**Conclusion**
- How well is the performance of the model?
- Is there any extra info that could help the model?
- In which criteria or use case, does the model provide more benefit can other statistical model?


-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
