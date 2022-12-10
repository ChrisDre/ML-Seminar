import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# load variable list from excel sheet
df_vars = pd.read_excel("variables.xlsx", sheet_name="categories")

# get 'income' column and remove NAVs
income_vars = df_vars["Income Vars"].dropna().unique().tolist()

# load gss data from 2018
df = pd.read_pickle("gss_filtered.pkl")

# filter dataframe for chosen variables
df = df[income_vars]

# check amount of nan values
df.isna().sum()

# drop all rows with missing income answers
df = df.dropna(subset=["income"])

# hrs1 is zero for all wrkstat categories except 1 and 2 (full time & part time)


# change 'sex' column to 'female' column
df['female'] = df['sex'].map({2: 1, 1: 0})
df = df.drop('sex', axis=1)





test = "stop"