
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title('Cardlytics')

def analyze_industry(industry):
	xcol = industry+' Incentive'
	ycol = industry+' Redemption'
	correlation = df[[xcol, ycol]].corr().values[0][1]
	reg = LinearRegression(fit_intercept=False).fit(df[[xcol]], df[[ycol]])
	coefficient = reg.coef_.item()
	st.header(industry)
	fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
	plt.tight_layout(pad=1, w_pad=3.5, h_pad=1.0)
	ax1.tick_params(axis = "x", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE [0,1] LABELS
	ax1.tick_params(axis = "y", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE [0,1] LABELS
	ax1 = fig.add_subplot(121)
	ax1.set_xlim(0, df[xcol].max()*1.02)
	ax1.set_ylim(0, df[ycol].max()*1.02)
	ax1.set_title('Redemption ~ {0:.3f}*Incentive (correlation: {1:.3f})'.format(coefficient, correlation))
	ax1 = sns.regplot(x=xcol, y=ycol, data=df[[xcol, ycol]])
	ax1.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)
	ax2.tick_params(axis = "x", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE LABELS
	ax2.tick_params(axis = "y", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE [0,1] LABELS
	ax2 = fig.add_subplot(122)
	for tick in ax2.get_xticklabels():
	    tick.set_rotation(90)
	ax2.plot(df.Date, df[xcol], color='orange')
	ax2.plot(df.Date, df[ycol], color='red') # , alpha=0.8	
	ax2.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)	
	ax2.legend(["Incentive", "Redemption"],loc="upper right")
	st.pyplot(fig)
	
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
	df = pd.read_csv(uploaded_file)
	cols = df.columns
	incentives = [c for c in cols if c.lower().find('incentive')>=0]
	redemptions = [c for c in cols if c.lower().find('redemption')>=0]
	industries = [c.replace(" Incentive", "") for c in incentives]
	sorted_industries = sorted(industries)
	selected_industries = st.sidebar.multiselect('Choose industries:', sorted_industries, sorted_industries)
	selected_columns = ['Date']+[i+' Incentive' for i in selected_industries]+[i+' Redemption' for i in selected_industries]
	st.write(df[selected_columns])
	analyze = st.checkbox('Display incentive-redemption relation for selected industries')
	if analyze and len(selected_industries)>0:
		for i in selected_industries:
			analyze_industry(i)

