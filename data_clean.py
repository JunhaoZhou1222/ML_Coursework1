import pandas as pd

path_train = 'CW1_train.csv'
path_test = 'CW1_test.csv'

df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

# cleaning the training data carat, depth, table, x, y, z
# ------------------remove outliers based on carat values using logic----------------
carat_outliers = df_train[(df_train['carat'] <= 0) | (df_train['carat'] > 5)].index
df_train.drop(carat_outliers, inplace=True)
print(f'Removed {len(carat_outliers)} outliers based on carat values.')

# ------------------remove outliers based on depth values using IQR------------------
depth_Q1 = df_train["depth"].quantile(0.25)
depth_Q3 = df_train["depth"].quantile(0.75)
depth_IQR = depth_Q3 - depth_Q1
depth_lower_bound = depth_Q1 - 1.5 * depth_IQR
depth_upper_bound = depth_Q3 + 1.5 * depth_IQR
depth_outliers = df_train[(df_train['depth'] < depth_lower_bound) | (df_train['depth'] > depth_upper_bound)].index
df_train.drop(depth_outliers, inplace=True)
print(f'Removed {len(depth_outliers)} outliers based on depth values.')

# ------------------remove outliers based on table values using IQR------------------
table_Q1 = df_train["table"].quantile(0.25)
table_Q3 = df_train["table"].quantile(0.75)
table_IQR = table_Q3 - table_Q1
table_lower_bound = table_Q1 - 1.5 * table_IQR
table_upper_bound = table_Q3 + 1.5 * table_IQR
table_outliers = df_train[(df_train['table'] < table_lower_bound) | (df_train['table'] > table_upper_bound)].index
df_train.drop(table_outliers, inplace=True)
print(f'Removed {len(table_outliers)} outliers based on table values.')

# ------------------remove outliers based on x, y, z values using logic------------------
x_outliers = df_train[(df_train['x'] <= 0)].index
df_train.drop(x_outliers,inplace=True)

y_outliers = df_train[(df_train['y'] <= 0)].index
df_train.drop(y_outliers,inplace=True)

z_outliers = df_train[(df_train['z'] <= 0)].index
df_train.drop(z_outliers,inplace=True)
print(f'Removed {len(x_outliers) + len(y_outliers) + len(z_outliers)} outliers based on dimensions x, y, z.')

#print(df_train.info())

# ------------------transfer str type of cut, color, categories to int---------------------
num_cut_categories = df_train["cut"].unique()
print(f'cut has {num_cut_categories}')
cut_map = {"Ideal":5, "Premium":4, "Very Good":3, "Good":2, "Fair":1}
df_train["cut"] = df_train["cut"].map(cut_map)
df_test["cut"] = df_test["cut"].map(cut_map)

num_color_catrgories = df_train["color"].unique()
print(f"color has {num_color_catrgories}")
color_map = {"D":1, "E":2, "F":3, "G":4, "H":5, "I":6, "J": 7}
df_train["color"] = df_train["color"].map(color_map)
df_test["color"] = df_test["color"].map(color_map)

num_clarity_categories = df_train["clarity"].unique()
print(f"clarity has {num_clarity_categories}")
clarity_map = {"IF":1, "VVS2":2, "VVS1":3, "VS2":4, "VS1":5, "SI2":6, "SI1":7, "I1":8}
df_train["clarity"] = df_train["clarity"].map(clarity_map)
df_test["clarity"] = df_test["clarity"].map(clarity_map)


df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
print(df_train.info())
print(df_train.head())

# save cleaned data csv files
df_train.to_csv('CW1_train_cleaned.csv', index=False)
df_test.to_csv('CW1_test_cleaned.csv', index=False)




