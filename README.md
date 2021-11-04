# HedgeWithSentiment

* Data
  * option data: OptionMetrics from WRDS
  * stock price data : CRSP from WRDS (not ajusted) or from yahoo finance (yfinance package)
  * It was put outside the code folder.  eg. './data/'
  
* preprocess  (all most the same except for dealing with input)
  * option_preprocess.py
  * option_preprocess_SPX.py
  * option_preprocess_industry.py 

* prediction
  * spx_forecast.py
  * spx_forecast_rnn.py
  * stock_forecast.py   

* models : folder core    

* main.ipynb (middle results, plots, etc.)
 
