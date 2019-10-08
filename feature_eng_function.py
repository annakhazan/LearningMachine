import pandas as pd
import numpy as np

def feature_eng_forest(data_file_path, soil_file_path, test=False):
	try:
		df = pd.read_csv(data_file_path, sep=',', header=0, index_col='Id')
		soil_types = pd.read_csv(soil_file_path).set_index('Soil Type')
	except:
		df = pd.read_csv(data_file_path, sep=',', header=0)
		soil_types = pd.read_csv(soil_file_path).set_index('Soil Type')
	def labelSoilType(row):
		for i in range(len(row)):
			if row[i] == 1:
				return 'Soil_Type'+str(i)
	df['Direct_Distance_To_Hydrology']=np.sqrt((df.Vertical_Distance_To_Hydrology**2) + (df.Horizontal_Distance_To_Hydrology**2)).astype(float).round(2)
	def azimuth_to_abs(x):
		if x>180:
			return 360-x
		else:
			return x

    # Create Soil Type Buckets
	soil_types = pd.read_csv('soil_types.csv').set_index('Soil Type')
	df['Soil Type'] = df[['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
           'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7',
           'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
           'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
           'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
           'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
           'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
           'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
           'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
           'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
           'Soil_Type40']].apply(lambda row: labelSoilType(row), axis=1)
	df = pd.merge(df, soil_types, how='left', left_on='Soil Type', right_index=True)
	del df['Soil Type'] # Delete string column

	# Create feature to that transforms azimuth to its absolute value
	df['Aspect2'] = df.Aspect.map(azimuth_to_abs)
	df['Aspect2'].astype(int)

	# Create feature that determines if the patch is above sea level
	df['Above_Sealevel'] = (df.Vertical_Distance_To_Hydrology>0).astype(int)

	# Bin the Elevation Feature: check the feature exploration notebook for motivation
	bins = [0, 2600, 3100, 8000]
	group_names = [1, 2, 3]
	df['Elevation_Bucket'] = pd.cut(df['Elevation'], bins, labels=group_names)
	df['Elevation_0_2600'] = np.where(df['Elevation_Bucket']== 1, 1, 0)
	df['Elevation_2600_3100'] = np.where(df['Elevation_Bucket']== 2, 1, 0)
	df['Elevation_3100_8000'] = np.where(df['Elevation_Bucket']== 3, 1, 0)
	df['Elevation_0_2600'].astype(int)
	df['Elevation_2600_3100'].astype(int)
	df['Elevation_3100_8000'].astype(int)
	del df['Elevation_Bucket']

	# Create a feature for no hillshade at 3pm
	df['3PM_0_Hillshade'] = (df.Hillshade_3pm == 0).astype(int)

	#Direct distance to hydrology
	df['Direct_Distance_To_Hydrology'] = np.sqrt((df.Vertical_Distance_To_Hydrology**2) + \
        (df.Horizontal_Distance_To_Hydrology**2)).astype(float).round(2)


	soil_types= ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
           'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
           'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
           'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
           'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
           'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
           'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
           'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
           'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
           'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
           'Soil_Type40', 'Cover_Type']

	column_list = df.columns.tolist()
	column_list = [c for c in column_list if c[:9] != 'Soil_Type']
	column_list.insert(10, 'Direct_Distance_To_Hydrology')
	column_list.insert(11, 'Elevation_0_2600')
	column_list.insert(12, 'Elevation_2600_3100')
	column_list.insert(13, 'Elevation_3100_8000')
	column_list.insert(14, 'Aspect2')
	column_list.insert(15, 'Above_Sealevel')
	column_list.insert(16, '3PM_0_Hillshade')
	column_list.extend(soil_types)
	columns = []
	for col in column_list:
		if col not in columns:
			if col != 'Cover_Type':
				columns.append(col)
	if test:
		pass
	else:
		columns.append('Cover_Type')

	df = df[columns]
	df.fillna(0,inplace=True) # Replace nans with 0 for our soil type bins

	if not test:    
		to_remove = [] # features to drop
		for c in df.columns.tolist():
			if df[c].std() == 0:
				to_remove.append(c)
		df = df.drop(to_remove, 1)
		print("Dropped the following columns: \n")
		for r in to_remove:
			print (r)
        
	return df

def forest_interactions(df, test=False): 
	if test:
		for i in range(df.shape[1]):
			for j in range(54):
				if i != j:
					df[df.columns.tolist()[i]+"_"+df.columns.tolist()[j]] = df[df.columns.tolist()[i]]*df[df.columns.tolist()[j]]
	else:
		for i in range(df.shape[1]-1):
			for j in range(54):
				if i != j:
					df[df.columns.tolist()[i]+"_"+df.columns.tolist()[j]] = df[df.columns.tolist()[i]]*df[df.columns.tolist()[j]]

	if not test:    
		to_remove = [] # features to drop
		for c in df.columns.tolist():
			if df[c].std() == 0:
				to_remove.append(c)
		df = df.drop(to_remove, 1)
		print("Dropped the following columns: \n")
		for r in to_remove:
			print (r)
            
	return df
