import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn import linear_model, tree
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.decomposition import FactorAnalysis 
from sklearn.ensemble import ExtraTreesClassifier




# load variable list from excel sheet
df_vars = pd.read_excel("variables.xlsx", sheet_name="categories")

# get chosen 'mental health' variables
chosen_vars = df_vars["Variable"].dropna().unique().tolist()

# load gss data from 2018
df = pd.read_pickle("gss_2018_no_labels.pkl")
#df = pd.read_pickle("gss_2018_labels.pkl")

# filter for mental vars
df = df[chosen_vars]


# load NAICs to classify industries
ind_codes_xl = pd.read_excel("industry_codes.xlsx")
ind_codes = dict(zip(ind_codes_xl["subsector"].to_list(), ind_codes_xl["id"].to_list()))


# preprocessing with labels
def preprocessing(df):

    # drop "year" column
    df.drop("year", axis=1, inplace=True)

    # remove rows with missing vote in 2016 (1 = Clinton, 2 = Trump, 3 = other candidate)
    df = df[df.pres16.isin([1, 2, 3])]

    # pres16: transform to trump var
    df['trump'] = np.where(df['pres16'] == 2, 1, 0)
    df.drop("pres16", axis=1, inplace=True)


    # replace missing values in hrs1 with hrs2 or zero for retired/unemp etc.
    df["hours_worked"] = df['hrs1'].where(~df["wrkstat"].isin([4, 5, 6, 7, 8]), 0)
    df["hours_worked"] = np.where(df["wrkstat"] == 3, df["hrs2"], df["hours_worked"])
    df.dropna(subset=["hours_worked"], inplace=True)
    df.drop("hrs1", axis=1, inplace=True)
    df.drop("hrs2", axis=1, inplace=True)

    # wrkstat
    df["wrkstat"] = df["wrkstat"].astype("category")

    # wrkslf: if missing but retired/unemp etc. set to not self employed
    df["selfemp"] = df["wrkslf"].map({1: 1, 2: 0})
    df["selfemp"] = df["selfemp"].fillna(0)
    df.drop("wrkslf", axis=1, inplace=True)

    # fill missing "age" cells with mean (4 missing)
    df["age"].fillna(df["age"].mean(), inplace=True)

    # "sex" column -> transform to "female" column
    df["female"] = df["sex"].map({2: 1, 1: 0})
    df.drop("sex", axis=1, inplace=True)

    # satjob: new category: 0 for retired/unemp etc. or no answer
    df["satjob"].fillna(0, inplace=True)
    df["satjob"] = df["satjob"].astype("category")

    # wrktype: new category: 0 for retired/unemp etc. or no answer
    df["wrktype"].fillna(0, inplace=True)
    df["wrktype"] = df["wrktype"].astype("category")

    # drop na rows
    df.dropna(inplace=True)

    # marital
    df["marital"] = df["marital"].astype("category")

    # partyid
    df["partyid"] = df["partyid"].astype("category")

    # relig: nothing
    df["relig"] = df["relig"].astype("category")

    # income16
    df["income16"] = df["income16"].astype("category")

    # wrkgovt, transform, remove missing later
    df["govemp"] = df["wrkgovt"].map({1: 1, 2: 0})
    df.drop("wrkgovt", axis=1, inplace=True)


    for col in ["natsoc", "natmass", "natpark", "natchld", "natsci", "natenrgy"]:
        df[col] = df[col].astype("category")


    # indus10
    df["industry"] = df["indus10"].map(ind_codes)
    df["industry"] = df["industry"].astype("category")
    df.drop("indus10", axis=1, inplace=True)

    # hispanic: 1 = not hispanic, else hispanic
    df['hispanic'] = np.where(df['hispanic'] == 1, 0, 1)

    # capppun -> transform to favor death penalty
    df["fav_death_pen"] = df["cappun"].map({1: 1, 2: 0})
    df.drop("cappun", axis=1, inplace=True)

    # vote12
    df['voted_12'] = np.where(df['vote12'] == 1, 1, 0)
    df.drop("vote12", axis=1, inplace=True)

    # vote16
    df['voted_16'] = np.where(df['vote16'] == 1, 1, 0)
    df.drop("vote16", axis=1, inplace=True)

    # TODO: transform categories to categorical
    # -> ordinal: consecutive integers (higher is better)
    # -> nominal: OHE

    return df


    # wrktype




# analytical base table (abt)
abt = df.copy()

abt = preprocessing(abt)


abt.info()

## Models

# create target column
y = abt["trump"]

# create feature colummns
X = abt.drop(["trump", "id"], axis=1)

# OHE
X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1111
)


# mental
# logistic
logisticRegr = linear_model.LogisticRegression()
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)


# The coefficients
print("Mental: Coefficients: \n", logisticRegr.coef_)
# The mean squared error
print("Mental: Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Mental: Coefficient of determination: %.2f" % r2_score(y_test, y_pred))





# trump
# logistic
# logisticRegr = linear_model.LogisticRegression()
# logisticRegr.fit(X_train, y_train)
# ly_pred = logisticRegr.predict(X_test)


# The coefficients
# print("Trump: Coefficients: \n", reg.coef_)
# # The mean squared error
# print("Trump: Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print("Trump: Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# Lasso
reg = linear_model.LassoCV(cv=10, random_state=0)

# Train the model using the training sets
reg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = reg.predict(X_test)

# The coefficients
print("Lasso: Coefficients: \n", reg.coef_)
# The mean squared error
print("Lasso: Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Lasso: Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# Ridge
reg = linear_model.RidgeCV(cv=10)

# Train the model using the training sets
reg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = reg.predict(X_test)

# The coefficients
print("Ridge: Coefficients: \n", reg.coef_)
# The mean squared error
print("Ridge: Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Ridge: Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# Elastic Net
reg = linear_model.ElasticNetCV(cv=10, random_state=0)

# Train the model using the training sets
reg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = reg.predict(X_test)

# The coefficients
print("Elastic Net: Coefficients: \n", reg.coef_)
# The mean squared error
print("Elastic Net: Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Elastic Net: Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# Decision tree
reg = tree.DecisionTreeRegressor()
reg = reg.fit(X_train, y_train)
reg.predict(X_test)

# The mean squared error
print("DT: Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("DT: Coefficient of determination: %.2f" % r2_score(y_test, y_pred))



# k nearest neighbor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)







## ML SPEED RUN

# setup ML pipelines
pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestRegressor(random_state=1234)),
    "gb": make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=1234))}


grid = {
    "rf": {"randomforestregressor__n_estimators": [100, 200, 300]},
    "gb": {"gradientboostingregressor__n_estimators": [100, 200, 300]},
}

# create a blank dictionary to hold the models
fit_models = {}

print(f"Starting models...")
# loop through algos
for algo, pipeline in pipelines.items():
    # create new grid search cv class
    model = GridSearchCV(pipeline, grid[algo], n_jobs=-1, cv=10)
    # train model
    model.fit(X_train, y_train)
    # store results in dictionary
    fit_models[algo] = model

print(f"Models done.")


# evaluate performance
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    mse = mean_squared_error(y_test, yhat)
    r2 = r2_score(y_test, yhat)
    map = mean_absolute_percentage_error(y_test, yhat)
    print(f"Metrics for {algo}: MSE:{mse}, R2: {r2}, MAP: {map}")


x= "stop"