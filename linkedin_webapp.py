import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
os.chdir('C:\\Users\\nickb\\Python')
st.title("LinkedIn User Prediction Project")

s = pd.read_csv('social_media_usage.csv')

def clean_sm(x):
    return np.where(x==1, 1, 0)

ss = pd.DataFrame()
ss['sm_li'] = s.web1h
ss[['income', 'education','parent','married','female','age']] = s[['income', 'educ2','par','marital','gender','age']]
# Processing
ss['sm_li'] = clean_sm(ss['sm_li'])
ss['income'] = np.where(ss['income'] < 9, ss['income'], np.nan)
ss['education'] =  np.where(ss['education'] < 8, ss['education'], np.nan)
ss['age'] = np.where(ss['age'] < 98, ss['age'], np.nan)
ss.dropna(inplace = True)

# Create Model
y = ss['sm_li']
X = ss.drop('sm_li', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression(random_state=42, class_weight = 'balanced', max_iter = 500).fit(X_train, y_train)

col1, col2 = st.columns(2)
col3, col4= st.columns(2)
col5, col6 = st.columns(2)

with col1:
    inc = st.number_input('Income', value = 8)

with col2:
    edu = st.number_input('Education', value =7)
    
with col3:
    par = st.number_input('Parent', value =1)
    
with col4:
    mar = st.number_input('Married', value =1)

with col5:
    gen = st.number_input('Gender', value =2)
    
with col6:
    age = st.number_input('Age', value =42)
    
if st.button('Make Prediction'):
    prob = log_reg.predict_proba(np.array([inc,edu,par,mar,gen,age ]).reshape(1, -1))
    st.metric("Model Prediction (Probability of Being a LinkedIn User): ", str(np.round(prob[0][1]*100,2)) + '%')