import pandas as pd
import pickle

# get chosen variables as list
df_vars = pd.read_excel("variables.xlsx", sheet_name="all")
unique_vars = df_vars.Variable.unique()
gss_vars = unique_vars.tolist()

# get complete data set
df = pd.read_stata("GSS7218_R1.dta", convert_categoricals=False, columns=gss_vars)

# convert columns
df.columns = [x.lower() for x in df.columns]

# filter by year
df = df[df["year"] == 2018]

# save to pickle for fast loading
df.to_pickle("gss_filtered.pkl")
