# st_app.py
# 100471173
# May 2026

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# App Title
st.title("UK GDP Nowcasting App")
st.write("Use the slider and buttons below to see how business confidence affects GDP predictions!")

# loading the saved csv data we created in the notebook
df = pd.read_csv('uk_gdp_sentiment_data.csv', parse_dates=['DATE'], index_col='DATE')

# creating a lagged GDP column needed for the model
df['GDP_Lagged'] = df['GDP_Growth'].shift(1)
df = df.dropna()

# radio button to switch between normal times and supply shock
# this reflects the tale of two crises finding from Ashwin et al. 2024
state = st.radio('Select Economic State:', ['Normal Times', 'Supply Shock'])

# the base sentiment coefficient from our regression
base_coefficient = 0.1088

# if supply shock is selected the coefficient increases by 50%
# this shows sentiment has a bigger impact during crises
if state == 'Normal Times':
    coefficient = base_coefficient
elif state == 'Supply Shock':
    coefficient = base_coefficient * 1.5

# slider to let the user adjust the sentiment value
sentiment_value = st.slider('Set Business Confidence Value:', min_value=-50.0, max_value=15.0, value=0.0, step=0.5)

# calculating the nowcast using our regression equation
# GDP = 1.1476 + (-0.7158 x last quarters GDP) + (coefficient x sentiment)
last_gdp = df['GDP_Lagged'].iloc[-1]
nowcast = 1.1476 + (-0.7158 * last_gdp) + (coefficient * sentiment_value)

# displaying the nowcast value using st.metric
st.metric(label='GDP Nowcast (%)', value=f'{nowcast:.2f}%')

# calculating fitted values using the selected coefficient
df['Nowcast'] = 1.1476 + (-0.7158 * df['GDP_Lagged']) + (coefficient * df['Sentiment'])

# creating the interactive plotly chart
fig = px.line(df, y=['GDP_Growth', 'Nowcast'],
              title=f'UK GDP Growth vs Nowcast ({state})',
              labels={'value': 'GDP Growth (%)', 'DATE': 'Year'})

# showing the chart in the app
st.plotly_chart(fig)