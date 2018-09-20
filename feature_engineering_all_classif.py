import pandas as pd
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from datetime import datetime
from datetime import timedelta
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# defining where the data is located
data_path = '/Users/ruoyangzhang/Documents/PythonWorkingDirectory/Eleven_super_case_customer_insights_Data/'

# identifying all the log files in the above defined file location
all_log_files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f[:3] == 'LOG']


# now loading in the non log data files and removing irrelevant columns and producing python objects to be used later

# conversion table
conver = pd.read_csv(data_path+'TABLE_CONVERSION_new.csv', sep = ';')
conver = conver.drop(['Unnamed: 0'], axis = 1)
conver.index = conver.CLIENT_NUMBER
# we only want to keep lines whose client number corresponds to one unique visitor ID)
con_cn_retain = [k for k,v in Counter(conver.CLIENT_NUMBER).items() if v == 1]
conver = conver.loc[con_cn_retain]
# create client number to visitor ID dict and its reverse
conver_dict = {row['CLIENT_NUMBER']:row['VISITOR_ID'] for i, row in tqdm(conver.iterrows())}
conver_dict_reverse = {v:k for k,v in conver_dict.items()}

# loading in the slimmed CMD data
CMD_slim = pd.read_csv(data_path+'CMD_slim.csv')
CMD_slim = CMD_slim.loc[:, ~CMD_slim.columns.str.contains('^Unname')]
CMD_slim.index = CMD_slim.identifiant

###################################### now we start treating the log files iteratively ###################################


for file in tqdm(all_log_files):
	# loading in the log file
	log_file = pd.read_csv(data_path+ file, sep=';')

	# we extract the period out of the file name
	period = file[8:14]
	period = period[:4] + '-' + period[-2:]

	# creating unique visit session (unique by session and by visitor)
	log_file['unique_visitor_session'] = [str(log_file.VISITOR_ID[i]) + str(log_file.ID_SESSION[i]) for i in range(len(log_file))]
	log_file.index = log_file.unique_visitor_session

	# identify the visits that lead to a purchase and slim the log file 
	idx_bought = log_file[['confirmation d achat' in str(page) for page in log_file.PAGES]].unique_visitor_session
	log_file = log_file.loc[idx_bought]
	log_file = log_file.reset_index(drop = True)

	# we then save this slimmed log file as a csv
	log_file.to_csv(data_path+'log_file_all_purchases_' + period + '.csv')

	# We further filter by retaining only the lines whose client numbers-visitor ID are found in the CMD file
	# load in the pre processed dictionary containing these client numbers
	with open(data_path+'client_by_date_net_all.pickle', 'rb') as handle:
		client_by_date_net_all = pickle.load(handle)

	visitor_cmd = [conver_dict[cn] for cn in client_by_date_net_all[period] if cn in conver_dict.keys()]
	visitor_id_to_retain = list(set(visitor_cmd)&set(log_file.VISITOR_ID))
	log_file.index = log_file.VISITOR_ID
	log_file = log_file.loc[visitor_id_to_retain]
	log_file = log_file.reset_index(drop = True)

	# now we start engineering the features
	log_file.index = log_file.unique_visitor_session
	unique_sessions = list(set(log_file.index))
	fields_unique_val = [ele for ele in list(log_file.columns) if ele not in ['EVENT_DATE', 'PAGES']]

	# we define a function to produce these features so that we can parallelise the process
	def return_dict_for_compression(unique_session_id):
	# this function takes the previously constructed unique session id (unique in the combination of visitor id and id session)
	# it produces the following features:
	# 1. time spent before the final page
	# 2. number of unique pages visited
	# 3. hour of visit
	# 4. landing page
	# 5. second page
		subdf = log_file.loc[unique_session_id]
		if isinstance(subdf, pd.DataFrame):
			dict_to_return = {field:list(set(subdf[field]))[0] for field in fields_unique_val}
			event_dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in subdf.EVENT_DATE]
			time_before_final_page = (max(event_dates) - min(event_dates)).total_seconds()
			sesh_start_hour = list(set(subdf.SESSION_START_DATE))[0][11:13]
			unique_page_count = len(set(subdf.PAGES))
			page_time_dict = {event_dates[i]: subdf.PAGES.iloc[i] for i in range(len(event_dates))}
			first_pg_time = min(event_dates)
			landing_page = page_time_dict[first_pg_time]
			event_dates.remove(min(event_dates))
			second_pg_time = min(event_dates)
			second_page = page_time_dict[second_pg_time]
		else:
			dict_to_return = {field:subdf[field] for field in fields_unique_val}
			time_before_final_page = 0
			sesh_start_hour = subdf.SESSION_START_DATE[11:13]
			unique_page_count = 1
			landing_page = subdf.PAGES
			second_page = np.nan
		dict_to_return['time_before_final'] = time_before_final_page
		dict_to_return['start_hour'] = sesh_start_hour
		dict_to_return['nb_pages_unique'] = unique_page_count
		dict_to_return['landing_page'] = landing_page
		dict_to_return['second_page'] = second_page
		return(dict_to_return)


	p = Pool(8, maxtasksperchild=1)
	compressed_lines = p.map(return_dict_for_compression, tqdm(unique_sessions))
	p.close()

	# we then turn the result into a dataframe
	visit_sessions = pd.DataFrame.from_dict(compressed_lines)
	visit_sessions['CLIENT_NUMBER'] = [conver_dict_reverse[vid] for vid in visit_sessions.VISITOR_ID]

	# align the information from the CMD file
	visit_sessions['identifiant'] = [str(row['CLIENT_NUMBER']) + row['SESSION_START_DATE'][:10] for i, row in tqdm(visit_sessions.iterrows())]
	visit_sessions.index = visit_sessions.identifiant
	idx_retain = list(set(visit_sessions.index)&set(CMD_slim.index))

	CMD_slim_2 = CMD_slim.loc[idx_retain]
	visit_sessions = visit_sessions.loc[idx_retain]
	visit_sessions_transac = pd.merge(left=visit_sessions, right=CMD_slim_2, on='identifiant', how='left')
	visit_sessions_transac.start_hour = [int(sh) for sh in visit_sessions_transac.start_hour]
	# we turn the start hour feature into 2 columns for cyclical representations so that we can use them in distance based clustering
	visit_sessions_transac['hour_sin'] = np.sin(2*np.pi*visit_sessions_transac.start_hour/24)
	visit_sessions_transac['hour_cos'] = np.cos(2*np.pi*visit_sessions_transac.start_hour/24)

	# isolate the numeric variables for clustering
	numeric_variables = ['nb_pages_unique', 'time_before_final','PRE_TAX_AMOUNT','hour_sin', 'hour_cos']
	visit_sessions_transac = visit_sessions_transac.reset_index(drop = True)

	# now cluster this baby!
	kmeans = KMeans(n_clusters=2, random_state=0).fit(visit_sessions_transac.loc[:, numeric_variables])
	# store the clusters in a new column
	visit_sessions_transac['cluster'] = kmeans.labels_

	# we then try to calculate the 'confusion matrix'
	cvic_binary_0 = [0 if cvic else 1 for cvic in visit_sessions_transac.CVIC]
	cvic_binary_1 = [1 if cvic else 0 for cvic in visit_sessions_transac.CVIC]

	cm = confusion_matrix(cvic_binary_0, visit_sessions_transac.cluster)
	# we save the cm and also print it in the terminal
	print(cm)
	with open(data_path+'cm' + period + '.pickle', 'wb') as handle:
		pickle.dump(cm, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print('confusion matrix for the period ', period, ' has been saved')

	################ label the data and save the bad boy ###############
	def assign_data_label(row):
		if row['CVIC']:
			if row['cluster'] == 0:
				return('catalogue_customer')
			else:
				return(np.nan)
		else:
			if row['cluster'] == 1:
				return('internet_customer')
			else:
				return(np.nan)
	visit_sessions_transac['label'] = [assign_data_label(row) for i,row in visit_sessions_transac.iterrows()]
	# we then save the new labeled set to csv
	visit_sessions_transac.to_csv(data_path + 'visit_sessions_transac_label_assigned_' + period + '.csv')
	print('period ', period, ' is complete')




