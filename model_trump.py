import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# load variable list from excel sheet
df_vars = pd.read_excel("variables.xlsx", sheet_name="categories")

# get chosen 'mental health' variables
chosen_vars = df_vars["Variable"].dropna().unique().tolist()

# load gss data from 2018
df = pd.read_pickle("gss_2018_no_labels.pkl")
# df = pd.read_pickle("gss_2018_labels.pkl")

# filter for mental vars
df = df[chosen_vars]


# load NAICs to classify industries
ind_codes_xl = pd.read_excel("industry_codes.xlsx")
ind_codes = dict(zip(ind_codes_xl["subsector"].to_list(), ind_codes_xl["id"].to_list()))


# preprocessing with labels
def preprocessing(df):

    # year: -> drop "year" column
    df.drop("year", axis=1, inplace=True)

    # id: -> do nothing

    # pres16: (Target Variable, 2016 presidential vote) -> remove rows with missing vote in 2016 (1 = Clinton, 2 = Trump, 3 = other candidate)
    df = df[df.pres16.isin([1, 2, 3])]

    # pres16: transform to binary
    df["trump"] = np.where(df["pres16"] == 2, 1, 0)
    df.drop("pres16", axis=1, inplace=True)

    # wrkstat: (Work Status) -> nominal categorical
    df["wrkstat"] = df["wrkstat"].astype("category")

    # hrs1: (Hours worked last week) -> replace missing values in hrs1 with hrs2 or zero for retired/unemp etc.
    df["hours_worked"] = df["hrs1"].where(~df["wrkstat"].isin([4, 5, 6, 7, 8]), 0)
    df["hours_worked"] = np.where(df["wrkstat"] == 3, df["hrs2"], df["hours_worked"])
    df.dropna(subset=["hours_worked"], inplace=True)
    df.drop("hrs1", axis=1, inplace=True)
    df.drop("hrs2", axis=1, inplace=True)

    # wrkslf: (Self employed or not) -> if missing but retired/unemp etc. set to not self employed
    df["selfemp"] = df["wrkslf"].map({1: 1, 2: 0})
    df["selfemp"] = df["selfemp"].fillna(0)
    df.drop("wrkslf", axis=1, inplace=True)

    # age: -> fill missing "age" cells with mean (4 missing)
    df["age"].fillna(df["age"].mean(), inplace=True)

    # sex: (gender) -> transform to binary "female" column
    df["female"] = df["sex"].map({2: 1, 1: 0})
    df.drop("sex", axis=1, inplace=True)

    # educ: (Education in years) -> do nothing

    # satjob: (job satisfaction) -> ordinal categorical
    # original order is 1 - Very satisfied to 4 - very dissatisfied -> reverse, because higher number means better
    satjob_mapper = {1: 4, 2: 3, 3: 2, 4: 1}
    df["jobsat"] = df["satjob"].replace(satjob_mapper)
    df.drop("satjob", axis=1, inplace=True)

    # wrktype: (type of work) -> nominal categorical
    df["wrktype"] = df["wrktype"].astype("category")

    # relig: (Religion) -> nominal categorical
    df["relig"] = df["relig"].astype("category")

    # wrkgovt: (Gov employee or not) -> transform to binary govemp column
    df["govemp"] = df["wrkgovt"].map({1: 1, 2: 0})
    df.drop("wrkgovt", axis=1, inplace=True)

    # prestg10: (prestige score of occupation) -> do nothing

    # indus10: (Industry of work) -> nominal, high cardinality, collapse to fewer with industry codes
    # reduces industries from 232 to 12
    df["industry"] = df["indus10"].map(ind_codes)
    df["industry"] = df["industry"].astype("category")
    df.drop("indus10", axis=1, inplace=True)

    # marital: (marriage status) -> nominal categorical
    df["marital"] = df["marital"].astype("category")

    # partyid: (political party affiliation) -> nominal categorical
    df["partyid"] = df["partyid"].astype("category")

    # vote12: (if voted in 2012) -> transform to binary voted_12 column
    df["voted_12"] = np.where(df["vote12"] == 1, 1, 0)
    df.drop("vote12", axis=1, inplace=True)

    # vote16: (same as vote12)
    df["voted_16"] = np.where(df["vote16"] == 1, 1, 0)
    df.drop("vote16", axis=1, inplace=True)

    # income16: (income category) -> ordinal categorical
    # TODO: check value counts and maybe collapse to fewer categories

    # nat variables: (opinion, if spending by govt is too little, about right or too much) -> nominal categorical
    for col in ["natsoc", "natmass", "natpark", "natchld", "natsci", "natenrgy"]:
        df[col] = df[col].astype("category")

    # capppun: (favor or oppose death pen) -> transform to binary fav_death_pen column
    df["fav_death_pen"] = df["cappun"].map({1: 1, 2: 0})
    df.drop("cappun", axis=1, inplace=True)

    # hispanic: (whether or not is hispanic) -> transform to binary column
    df["hispanic"] = np.where(df["hispanic"] == 1, 0, 1)

    # vetyears: (years in armed forces) -> do nothing

    # drop na rows
    df.dropna(inplace=True)

    return df


# analytical base table (abt)
abt = df.copy()

# run dataframe through preprocessing
abt = preprocessing(abt)

# show info
print(abt.info())

# TODO: visualization and summarization
# TODO: feature selection
# TODO: find classification algos

## Models

# create target column
y = abt["trump"]

# create feature colummns
X = abt.drop(["trump", "id"], axis=1)

# One Hot Encoding (create dummy vars for nominal categories), set drop_first=True to avoid dummy variable trap
X = pd.get_dummies(X, drop_first=True)

# create train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1111
)


# TODO: it is now a classifier problem
np.random.seed(5000)

# init list for comparison
models = []


def algo_models(X_train, X_test, y_train, y_test):

    models.append(("LR", LogisticRegression()))
    models.append(("CART", DecisionTreeClassifier()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("NB", GaussianNB()))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append(("SVM", SVC()))

    outcomes = []
    names = []

    # loop through models
    for name, model in models:
        v_results = cross_val_score(
            model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1, verbose=0
        )
        print(name, v_results.mean())
        outcomes.append(v_results)
        names.append(name)

    print(outcomes)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticklabels(names)
    plt.boxplot(outcomes)


algo_models(X_train, X_test, y_train, y_test)


# Evaluating and predicting models
for name, model in models:
    trainedmodel = model.fit(X_train, y_train)

    # prediction
    ypredict = trainedmodel.predict(X_test)

    acc = accuracy_score(y_test, ypredict)
    classreport = classification_report(y_test, ypredict)
    confMat = confusion_matrix(y_test, ypredict)

    print("\n****************************" + name)
    print("The accuracy: {}".format(acc))
    print("The Classification Report:\n {}".format(classreport))
    print("The Confusion Matrix:\n {}".format(confMat))


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
    "gb": make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=1234)),
}


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


x = "stop"
