import pandas as pd


raw_data_path = 'Raw Data.xlsx'
df = pd.read_csv(raw_data_path)
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=10)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_standardized), columns=df_standardized.columns)