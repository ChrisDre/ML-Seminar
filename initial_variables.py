import pandas as pd
import pickle


# get all data from 2018 without labels
df_no_labels = pd.read_stata("GSS7218_R1_2018.dta", convert_categoricals=False)

# get all data from 2018 with labels (dropped before: hhtype, ISCO08, PAISCO08, MAISCO08, SPISCO08)
df_labels = pd.read_stata("GSS7218_R1_2018.dta", convert_categoricals=True)

# convert columns
df_no_labels.columns = [str(x) for x in df_no_labels.columns]
df_no_labels.columns = [x.lower() for x in df_no_labels.columns]
df_labels.columns = [str(x) for x in df_labels.columns]
df_labels.columns = [x.lower() for x in df_labels.columns]

# save to pickle for fast loading
df_no_labels.to_pickle("gss_2018_no_labels.pkl")
df_labels.to_pickle("gss_2018_labels.pkl")
