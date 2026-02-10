import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

st.set_page_config(page_title="Diabetes Regression",page_icon="",layout="centered")
st.title(" Diabetes Progression Regression")

data=load_diabetes()
X_train,X_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

col1,col2=st.columns(2)
with col1:
    st.metric("Mean Squared Error",f"{mean_squared_error(y_test,y_pred):.2f}")
with col2:
    st.metric("R² Score",f"{r2_score(y_test,y_pred):.2f}")

fig,axs=plt.subplots(1,2,figsize=(12,5))
axs[0].scatter(y_test,y_pred,color="blue",alpha=0.6)
axs[0].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"k--",lw=2)
axs[0].set_title("True vs Predicted")
axs[0].set_xlabel("Actual")
axs[0].set_ylabel("Predicted")

axs[1].scatter(X_test[:,2],y_pred,color="green",alpha=0.7)
axs[1].set_title("BMI vs Predicted")
axs[1].set_xlabel("BMI Feature")
axs[1].set_ylabel("Predicted")

st.pyplot(fig)
st.caption("Linear Regression on Diabetes Dataset")
