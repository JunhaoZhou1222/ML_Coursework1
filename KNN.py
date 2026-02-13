import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Set seed
np.random.seed(123)

# Import training data
trn = pd.read_csv('CW1_train.csv')
X_tst = pd.read_csv('CW1_test.csv')

# cleaning the training data carat, depth, table, x, y, z
# ------------------remove outliers based on carat values using logic----------------
print("start cleaning data...")
carat_outliers = trn[(trn['carat'] <= 0) | (trn['carat'] > 5)].index
trn.drop(carat_outliers, inplace=True)
print(f'Removed {len(carat_outliers)} outliers based on carat values.')

# ------------------remove outliers based on depth values using IQR------------------
depth_Q1 = trn["depth"].quantile(0.25)
depth_Q3 = trn["depth"].quantile(0.75)
depth_IQR = depth_Q3 - depth_Q1
depth_lower_bound = depth_Q1 - 1.5 * depth_IQR
depth_upper_bound = depth_Q3 + 1.5 * depth_IQR
depth_outliers = trn[(trn['depth'] < depth_lower_bound) | (trn['depth'] > depth_upper_bound)].index
trn.drop(depth_outliers, inplace=True)
print(f'Removed {len(depth_outliers)} outliers based on depth values.')

# ------------------remove outliers based on table values using IQR------------------
table_Q1 = trn["table"].quantile(0.25)
table_Q3 = trn["table"].quantile(0.75)
table_IQR = table_Q3 - table_Q1
table_lower_bound = table_Q1 - 1.5 * table_IQR
table_upper_bound = table_Q3 + 1.5 * table_IQR
table_outliers = trn[(trn['table'] < table_lower_bound) | (trn['table'] > table_upper_bound)].index
trn.drop(table_outliers, inplace=True)
print(f'Removed {len(table_outliers)} outliers based on table values.')

# ------------------remove outliers based on x, y, z values using logic------------------
x_outliers = trn[(trn['x'] <= 0)].index
trn.drop(x_outliers,inplace=True)

y_outliers = trn[(trn['y'] <= 0)].index
trn.drop(y_outliers,inplace=True)

z_outliers = trn[(trn['z'] <= 0)].index
trn.drop(z_outliers,inplace=True)
print(f'Removed {len(x_outliers) + len(y_outliers) + len(z_outliers)} outliers based on dimensions x, y, z.')

#print(df_train.info())

# ------------------transfer str type of cut, color, categories to int---------------------
num_cut_categories = trn["cut"].unique()
print(f'cut has {num_cut_categories}')
cut_map = {"Ideal":5, "Premium":4, "Very Good":3, "Good":2, "Fair":1}
trn["cut"] = trn["cut"].map(cut_map)
X_tst["cut"] = X_tst["cut"].map(cut_map)

num_color_catrgories = trn["color"].unique()
print(f"color has {num_color_catrgories}")
color_map = {"D":1, "E":2, "F":3, "G":4, "H":5, "I":6, "J": 7}
trn["color"] = trn["color"].map(color_map)
X_tst["color"] = X_tst["color"].map(color_map)

num_clarity_categories = trn["clarity"].unique()
print(f"clarity has {num_clarity_categories}")
clarity_map = {"IF":1, "VVS2":2, "VVS1":3, "VS2":4, "VS1":5, "SI2":6, "SI1":7, "I1":8}
trn["clarity"] = trn["clarity"].map(clarity_map)
X_tst["clarity"] = X_tst["clarity"].map(clarity_map)

train_nans = trn.isnull().sum().sum()
test_nans = X_tst.isnull().sum().sum()

if train_nans == 0 and test_nans == 0:
    print("no nans in the cleaned data")
else:
    print(f"find nans in cleaned data, Train: {train_nans}, Test: {test_nans}")


print("data cleaning completed.")
print("start training model...")
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']

X_tst = X_tst[X_trn.columns]

X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_trn, y_trn, test_size=0.2, random_state=123)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sub)
X_val_scaled = scaler.transform(X_val)

model = KNeighborsRegressor(n_neighbors=10)
model.fit(X_train_scaled, y_train_sub)
pred = model.predict(X_val_scaled)
r2 = r2_score(y_val, pred)
print(f" KNN R^2: {r2:.5f}")




