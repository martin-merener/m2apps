import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title='Rates insights 🦡', page_icon='🐐', layout='centered') # 🦡 badger;  🐐 goat
symbols_df = pd.read_csv("symbols.csv")
st.markdown('''# Open-Close Price Change''')
st.sidebar.markdown('''### How do you want to get your data?''')
choice = st.sidebar.radio("", ('Typing symbols/company names', 'Exploring by sector', 'Exploring by industry', 'From a local file'))

selected_symbols=None
data=None

symbol_company_pairs = [tuple(a[0:2]) for a in symbols_df.values]
day_dict = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}


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
		first_dt = datetime.datetime.strptime(str(dates[0])[0:10], "%Y-%m-%d").date()
		last_dt = datetime.datetime.strptime(str(dates[-1])[0:10], "%Y-%m-%d").date()
		if np.busday_count(first_dt, last_dt)/data.shape[0]>0.5:
			freq = 'day'
		else:
			freq = 'hour'
		st.markdown("###### The data uploaded appears to have one observation per {0}.".format(freq))
		#hours = [datetime.datetime.strptime(dt, "%Y-%m-%d").hour for dt in dates]
		#st.write(np.std(np.array(hours)))
if choice!="From a local file" and selected_symbols:
	today = datetime.date.today()
	yesterday = today - datetime.timedelta(days=1)
	another_day = yesterday - relativedelta(months=3)
	start_date = st.sidebar.date_input('Start date', another_day) 
	end_date = st.sidebar.date_input('End date', yesterday)
	freq = st.sidebar.selectbox('Frequency', ['day','hour'])
	freq_d = {'day':'1d', 'hour':'1h'}	
	down = st.sidebar.checkbox('Download data')
	if down:
		data = yf.download(selected_symbols, start=start_date, end=end_date, interval = freq_d[freq])

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
				data_analyze[('change_pct', s)] = (data_analyze[('Close', s)] - data_analyze[('Open', s)]) / data_analyze[('Open', s)] * 100
			change_rates = data_analyze.copy()
			change_rates = change_rates[[('change_pct',s) for s in symbols]]
			change_rates.columns = symbols
			if freq=='day':
				change_rates['weekday'] = change_rates.index
				change_rates['weekday'] = change_rates['weekday'].apply(lambda dt: day_dict[datetime.datetime.strptime(str(dt)[0:10], "%Y-%m-%d").weekday()])
			else:
				change_rates['hour'] = change_rates.index
				change_rates['hour'] = change_rates['hour'].apply(lambda dt: datetime.datetime.strptime(str(dt)[0:19], "%Y-%m-%d %H:%M:%S").hour)
			if change_rates.shape[1]>0:
				st.write("here?")
				latex = r'''### Some insights on the price change (%), $r = \frac{S_{close}}{S_{open}}-1$'''
				st.markdown(latex)
				for s in symbols:
					st.markdown('''##### {0} - {1}:'''.format(s, dict(symbol_company_pairs)[s]))
					fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
					#plt.tight_layout(pad=1, w_pad=3.5, h_pad=1.0)
					ax1.tick_params(axis = "x", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE LABELS
					ax1.tick_params(axis = "y", which = "both", bottom = True, top = True, direction='in', labelcolor='white')
					ax1 = fig.add_subplot(121)
					ax1.set_title('Histogram & density for % change')
					ax1.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)
					ax1 = sns.histplot(change_rates[s], bins=50, kde=True)
					ax1.set_xlabel('% change')
					ax1.set_ylabel('Count')
					if freq=='day':
						ax2.set_title('% change by day of the week')				
						ax2.tick_params(axis = "x", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE LABELS
						ax2.tick_params(axis = "y", which = "both", bottom = True, top = True, direction='in', labelcolor='white')
						ax2 = fig.add_subplot(122)
						ax2.set_xlabel('Day of week')
						ax2.set_ylabel('% change')
						ax2 = sns.violinplot(x = list(change_rates['weekday']), y = np.array(change_rates[s]))
						ax2.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)	
						st.pyplot(fig)				
					else:
						ax2.set_title('% change by hour of the day')				
						ax2.tick_params(axis = "x", which = "both", bottom = True, top = True, direction='in', labelcolor='white') # COULD NOT FIND ANOTHER WAY TO HIDE THESE LABELS
						ax2.tick_params(axis = "y", which = "both", bottom = True, top = True, direction='in', labelcolor='white')
						ax2 = fig.add_subplot(122)
						ax2.set_xlabel('Hour of day')
						ax2.set_ylabel('% change')
						ax2 = sns.violinplot(x = list(change_rates['hour']), y = np.array(change_rates[s]))
						ax2.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.2)	
						st.pyplot(fig)

except:
	print("No data available")


#@st.cache
#def load_data():
#    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#    html = pd.read_html(url, header = 0)
#    df = html[0]
#    return df
#
#df = load_data()
#df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']].to_csv("symbols.csv", Index-False)

