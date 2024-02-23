import pandas as pd
import streamlit as st
import numpy as np

st.title('Intrusion Detection Application')

st.header('Model Selection')
model = st.radio('Choose IDS model:', ['Tree-Based IDS', 'MTH IDS', 'LCCDE IDS'], index=None)

parameters = {}
st.header('Training Parameters')
train_proportion = st.select_slider(label='Select the training size percentage (Train/Test Split):', options=[f'{val:.0f}%' for val in np.linspace(50, 95, 10)])

criterion = st.radio('Choose criterion:', ['Gini', 'Entropy'], index=None)
parameters['criterion'] = criterion

lr = st.select_slider(label='Select learning rate:', options=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 0.9])
parameters['learning rate'] = lr
