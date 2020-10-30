#!/usr/bin/env python
# coding: utf-8

# In[3]:


import yfinance as yf
import streamlit as st
import pandas as pd
import datetime
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
import plotly.figure_factory as ff
from plotly import graph_objs as go
import warnings

warnings.filterwarnings("ignore") 

ticker = pd.Series(index=['State Bank of India','ITC Limited','ICICI Bank Limited','Reliance Industries Limited','Nestle India Limited'],
                   data=['SBIN.NS', 'ITC.NS', 'ICICIBANK.NS','RELIANCE.NS','NESTLEIND.NS'])

st.sidebar.header('User Input')
df = st.sidebar.selectbox('Select the stock', ticker.index)

today = datetime.date.today()
fiveyear = today - datetime.timedelta(days=1825)
start_date = st.sidebar.date_input('Start date', fiveyear)
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')
    
st.title("""
# Stock Price Prediction App 
""")
    

#define the ticker symbol
tickerSymbol = ticker[df]
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)
# Open	High	Low	Close	Volume	Dividends	Stock Splits

tickerDf = tickerDf.reset_index()

# Select only the important features i.e. the date and price
data = tickerDf[["Date","Close"]]

# Rename the features: These names are needed for the model fitting
data = data.rename(columns = {"Date":"ds","Close":"y"})


m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(data) # fit the model using all data
future = m.make_future_dataframe(periods=60) #we need to specify the number of days in future
prediction = m.predict(future)


#plot forecast
fig1 = plot_plotly(m, prediction,xlabel='Date',ylabel='Share price')
st.write('Forecasting closing of stock value for '+df)
st.plotly_chart(fig1)

#plot component wise forecast
st.write("Component wise forecast")
fig2 = m.plot_components(prediction)
# st.write(fig2)
st.plotly_chart(fig2)

