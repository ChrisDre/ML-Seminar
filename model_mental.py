import pickle
import pandas as pd


# load variable list from excel sheet
df_vars = pd.read_excel("variables.xlsx", sheet_name="categories")

# get 'mental health' column and remove NAVs
mental_vars = df_vars["Mental Health Vars"].dropna().unique().tolist()

# load gss data from 2018
df = pd.read_pickle("gss_filtered.pkl")

# filter dataframe for chosen variables
df = df[mental_vars]

# check amount of nan values
print(df.isna().sum())

# drop all rows with missing income answers -> question is mental health for working population, others didnt get the question
df = df.dropna(subset=["mntlhlth"])

# drop all rows with missing age answers
df = df.dropna(subset=["age"])

# drop all rows with missing educ answers
df = df.dropna(subset=["educ"])

# drop all rows with missing educ answers
df = df.dropna(subset=["relig"])

# drop all rows with missing wrktype answers
df = df.dropna(subset=["wrktype"])

# drop all rows with missing income answers
df = df.dropna(subset=["income"])

# drop all rows with missing wrkgovt answers
df = df.dropna(subset=["wrkgovt"])

# drop all rows with missing indus10 answers
df = df.dropna(subset=["indus10"])

# most missing values are wrkstat 3 (wtih job but ill, vacation or strike) -> take hrs2 for these (usually work)
df.hrs1[df.hrs1.isnull()] = df.hrs2

# drop hrs2 column (not needed anymore now)
df = df.drop('hrs2', axis=1)


# 3 are full time and no answer -> drop





# change 'sex' column to 'female' column
df['female'] = df['sex'].map({2: 1, 1: 0})
df = df.drop('sex', axis=1)




