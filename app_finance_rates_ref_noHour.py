import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from dateutil.relativedelta import relativedelta

def choose_reference(start_date, end_date):
	reference = st.sidebar.selectbox('Index reference', ['', 'Equities', 'Rates', 'Commodities', 'Volatility', 'Currencies'])
	index_d = {'Equities':'SPY', 'Rates':'^TNX', 'Commodities':'GC=F', 'Volatility':'^VIX', 'Currencies':'EURUSD=X'}
	ref_index = index_d.get(reference, "")
	if len(ref_index)>0:
		index_tickerData = yf.Ticker(ref_index)
		index_tickerDF = index_tickerData.history(interval='1d', start=start_date, end=end_date)
		index_tickerDF[ref_index] = (index_tickerDF['Close'] - index_tickerDF['Open']) / index_tickerDF['Open']*100
		index_tickerDF[reference] = (index_tickerDF['Close'] - index_tickerDF['Open']) / index_tickerDF['Open']*100
		return index_tickerDF

#st.set_page_config(page_title='Rates insights 🦡', page_icon='🐐', layout='centered') # 🦡 badger;  🐐 goat
st.set_page_config(page_title='Rates insights 🦡', page_icon='🐐', layout='wide') # 🦡 badger;  🐐 goat
symbols_df = pd.read_csv("symbols.csv")
st.markdown('''# Open-Close Price Change''')
st.sidebar.markdown('''### How do you want to get your data?''')
choice = st.sidebar.radio("", ('Typing symbols/company names', 'Exploring by sector', 'Exploring by industry', 'From a local file'))

selected_symbols=None
data=None

symbol_company_pairs = [tuple(a[0:2]) for a in symbols_df.values]
day_dict = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 11:'Dec'}
qrt_dict = {1:'Q1', 2:'Q2', 3:'Q3', 4:'Q4'}
#freq_d = {'day':'1d', 'hour':'1h'}	



if choice=='Typing symbols/company names':
	selection = st.sidebar.multiselect('Type symbol or company name: ', symbol_company_pairs)
	selected_symbols = [p[0] for p in selection]
elif choice=='Exploring by sector':
	sectors = list(set(symbols_df['GICS Sector']))
	selection = st.sidebar.multiselect('Choose sectors: ', sectors)
	symbols_df_2 = symbols_df.loc[symbols_df['GICS Sector'].isin(selection)].copy()
	symbols_df_2.reset_index(drop=True, inplace=True)
	symbol_company_pairs = [tuple(a[0:2]) for a in symbols_df_2.values]
	if len(symbol_company_pairs)>0:
		selection = st.sidebar.multiselect('Type symbol or company name: ', symbol_company_pairs)
		selected_symbols = [p[0] for p in selection]
elif choice=='Exploring by industry':
	industries = list(set(symbols_df['GICS Sub-Industry']))
	selection = st.sidebar.multiselect('Choose industries: ', industries)
	symbols_df_2 = symbols_df.loc[symbols_df['GICS Sub-Industry'].isin(selection)].copy()
	symbols_df_2.reset_index(drop=True, inplace=True)
	symbol_company_pairs = [tuple(a[0:2]) for a in symbols_df_2.values]
	if len(symbol_company_pairs)>0:
		selection = st.sidebar.multiselect('Type symbol or company name: ', symbol_company_pairs)
		selected_symbols = [p[0] for p in selection]
elif choice=='From a local file':
	uploaded_file = st.sidebar.file_uploader("")
	if uploaded_file is not None:
		data = pd.read_csv(uploaded_file, header=[0,1], index_col=0)
		dates = sorted(list(data.index))
		start_date = datetime.datetime.strptime(str(dates[0])[0:10], "%Y-%m-%d").date()
		end_date = datetime.datetime.strptime(str(dates[-1])[0:10], "%Y-%m-%d").date()
		index_tickerDF = choose_reference(start_date, end_date)

if choice!="From a local file" and selected_symbols:
	today = datetime.date.today()
	yesterday = today - datetime.timedelta(days=1)
	another_day = yesterday - relativedelta(months=3)
	start_date = st.sidebar.date_input('Start date', another_day) 
	end_date = st.sidebar.date_input('End date', yesterday)
	index_tickerDF = choose_reference(start_date, end_date)
	down = st.sidebar.checkbox('Download data')
	if down:
		data = yf.download(selected_symbols, start=start_date, end=end_date, interval = '1d')
		if len(selected_symbols)==1:
			data.columns=pd.MultiIndex.from_tuples([(v,selected_symbols[0]) for v in list(data.columns)]) # a patch because the data columns are different if there's only 1 symbol
try:
	if len(data)>0:
		st.write("Input data:")
		st.write(data)
		data.sort_index(inplace=True)
		missing_summary = pd.DataFrame(data.isna().sum())
		missing_summary.columns = ['# missing values found']
		if missing_summary['# missing values found'].sum()>0:
			st.markdown(''' ''')
			st.markdown('''##### Some values in your data were missing, and they were filled via interpolation.''')
			st.write(missing_summary.loc[missing_summary['# missing values found']>0])
			data.interpolate(inplace=True)
		variables = list(set([p[0] for p in data.columns]))
		symbols = list(set([p[1] for p in data.columns]))
		if ('Close' in variables) and ('Open' in variables):
			cols_analyze = [(v,s) for v in ['Open', 'Close'] for s in symbols]
			data_analyze = data[cols_analyze].copy()
			for s in symbols:
				data_analyze[('change_pct', s)] = (data_analyze[('Close', s)] - data_analyze[('Open', s)]) / data_analyze[('Open', s)]*100
			change_rates = data_analyze.copy()
			change_rates = change_rates[[('change_pct',s) for s in symbols]]
			change_rates.columns = symbols
			change_rates['weekday'] = change_rates.index
			change_rates['weekday'] = change_rates['weekday'].apply(lambda dt: datetime.datetime.strptime(str(dt)[0:10], "%Y-%m-%d").weekday())
			change_rates['month'] = change_rates.index
			change_rates['month'] = change_rates['month'].apply(lambda dt: datetime.datetime.strptime(str(dt)[0:10], "%Y-%m-%d").month)
			change_rates['quarter'] = change_rates['month'].apply(lambda m: int(np.ceil(m/3)))				
			if change_rates.shape[1]>0:
				latex = r'''## Insights on % price change: $r = \frac{S_{close}}{S_{open}}-1$'''
				st.markdown(latex)
				for s in symbols:
					st.markdown('''#### {0} - {1}:'''.format(s, dict(symbol_company_pairs)[s]))
					#if freq=='day':
					fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(12,3))
					plt.tight_layout(pad=1, w_pad=3.5, h_pad=1.0)
					ax1.set_title('Histogram for % change')
					ax1.tick_params(axis = "x", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE LABELS
					ax1.tick_params(axis = "y", which = "both", bottom = True, top = True, direction='in', labelcolor='white')
					ax1 = fig.add_subplot(141)
					ax1.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)
					ax1 = sns.histplot(change_rates[s], bins=50, kde=True)
					ax1.set_xlabel('% change')
					ax1.set_ylabel('Count')

					change_rates_s = change_rates.sort_values(by='weekday').copy()
					change_rates_s['weekday'] = change_rates_s['weekday'].apply(lambda x: day_dict[x])
					ax2.set_title('% change by day of the week')				
					ax2.tick_params(axis = "x", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE LABELS
					ax2.tick_params(axis = "y", which = "both", bottom = True, top = True, direction='in', labelcolor='white')
					ax2 = fig.add_subplot(142)
					ax2.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)	
					ax2 = sns.violinplot(x = change_rates_s['weekday'], y = np.array(change_rates[s]))
					ax2.set_xlabel('Day')
					ax2.set_ylabel('% change')

					change_rates_s = change_rates.sort_values(by='month').copy()
					change_rates_s['month'] = change_rates_s['month'].apply(lambda x: month_dict[x])
					ax3.set_title('% change by month')
					ax3.tick_params(axis = "x", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE LABELS
					ax3.tick_params(axis = "y", which = "both", bottom = True, top = True, direction='in', labelcolor='white')
					ax3 = fig.add_subplot(143)
					for tick in ax3.get_xticklabels():
					    tick.set_rotation(90)						
					ax3.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)	
					ax3 = sns.violinplot(x = change_rates_s['month'], y = np.array(change_rates[s]))
					ax3.set_xlabel('Month')
					ax3.set_ylabel('% change')

					change_rates_s = change_rates.sort_values(by='quarter').copy()
					change_rates_s['quarter'] = change_rates_s['quarter'].apply(lambda x: qrt_dict[x])
					ax4.set_title('% change by quarter')
					ax4.tick_params(axis = "x", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE LABELS
					ax4.tick_params(axis = "y", which = "both", bottom = True, top = True, direction='in', labelcolor='white')
					ax4 = fig.add_subplot(144)
					ax4.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)	
					ax4 = sns.violinplot(x = change_rates_s['quarter'], y = np.array(change_rates[s]))
					ax4.set_xlabel('quarter')
					ax4.set_ylabel('% change')

					st.pyplot(fig)

					try:
						if index_tickerDF.shape[0]>0:
							ref_index = index_tickerDF.columns[-2]
							reference = index_tickerDF.columns[-1]						
							
							st.markdown('''#### Comparing {0} to {1} index:'''.format(s, reference))
							change_rates['date'] = change_rates.index
							change_rates['date'] = change_rates['date'].apply(lambda dt: str(dt)[0:10])
							change_rates.set_index('date', inplace=True)

							index_tickerDF['date'] = index_tickerDF.index
							index_tickerDF['date'] = index_tickerDF['date'].apply(lambda dt: str(dt)[0:10])
							index_tickerDF.set_index('date', inplace=True)
							change_rates_and_reference = change_rates.merge(index_tickerDF, left_index=True, right_index=True)

							fig, ax1 = plt.subplots(figsize=(1,1))
							ax1 = fig.add_subplot(111)
							ax1 = sns.jointplot(data=change_rates_and_reference, x=ref_index, y=s, kind='reg')
							st.pyplot(ax1)

							# OR:

							# fig = plt.figure(figsize=(4,3))
							# ax = plt.axes()
							# ax = sns.jointplot(data=change_rates_and_reference, x=ref_index, y=s, kind='reg')
							# st.pyplot(ax)

							# OR:

							# fig = sns.jointplot(data=change_rates_and_reference, x=ref_index, y=s, kind='reg')
							# sns.set(rc={'figure.figsize':(4,4)})
							# st.pyplot(fig)
					except:
						st.write("(No index reference provided)")

except:
	print("No data yet")


#@st.cache
#def load_data():
#    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#    html = pd.read_html(url, header = 0)
#    df = html[0]
#    return df
#
#df = load_data()
#df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']].to_csv("symbols.csv", Index-False)

