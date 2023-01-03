import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import math


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.feature_selection import SelectFromModel




from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# set seed
np.random.seed(5000)

# load variable list from excel sheet
df_vars = pd.read_excel("variables.xlsx", sheet_name="categories")

# get chosen variables
chosen_vars = df_vars["Variable"].dropna().unique().tolist()

# load gss survey data from 2018 (already pre-filtered for easier import)
# use some with labels for easier processing)
df = pd.read_pickle("gss_2018_no_labels.pkl")
df_labels = pd.read_pickle("gss_2018_labels.pkl")

# filter for trump vars
df = df[chosen_vars]
df_labels = df_labels[chosen_vars]

# replace some variables with labeled data
vars_with_labels = ["wrkstat", "relig", "marital", "partyid", "natsoc", "natmass", "natpark", "natchld", "natsci", "natenrgy"]
df[vars_with_labels] = df_labels[vars_with_labels]

# print first 5 rows and dimensions of unprocessed dataframe
print(f"Unprocessed data - First rows: \n {df.head()}")
print(f"Unprocessed data - Dimensions: {df.shape}")

# load NAICs to classify industries
ind_codes_xl = pd.read_excel("industry_codes.xlsx")
#ind_codes = dict(zip(ind_codes_xl["subsector"].to_list(), ind_codes_xl["id"].to_list()))
ind_codes_labels = dict(zip(ind_codes_xl["subsector"].to_list(), ind_codes_xl["sector"].to_list()))


# preprocessing with labels
def preprocessing(raw_df):

    # reset index
    raw_df.reset_index(drop=True, inplace=True)

    # id: -> drop "id" column
    raw_df.drop("id", axis=1, inplace=True)

    # year: -> drop "year" column
    raw_df.drop("year", axis=1, inplace=True)

    # pres16: (Target Variable, 2016 presidential vote) -> remove rows with missing vote in 2016 (1 = Clinton, 2 = Trump, 3 = other candidate)
    raw_df = raw_df[raw_df.pres16.isin([1, 2, 3])].copy()

    # pres16: transform to binary
    raw_df["trump"] = np.where(raw_df["pres16"] == 2, 1, 0)
    raw_df.drop("pres16", axis=1, inplace=True)

    # wrkstat: (Work Status) -> do nothing

    # hrs1: (Hours worked last week) -> replace missing values in hrs1 with hrs2 or zero for retired/unemp etc.
    raw_df["hours_worked"] = raw_df["hrs1"].where(
        raw_df["wrkstat"].isin(["WORKING FULLTIME", "WORKING PARTTIME", "TEMP NOT WORKING"]), 0
    )
    raw_df["hours_worked"] = np.where(
        raw_df["wrkstat"] == "TEMP NOT WORKING", raw_df["hrs2"], raw_df["hours_worked"]
    )
    raw_df.dropna(subset=["hours_worked"], inplace=True)
    raw_df.drop("hrs1", axis=1, inplace=True)
    raw_df.drop("hrs2", axis=1, inplace=True)

    # wrkslf: (Self employed or not) -> if missing but retired/unemp etc. set to not self employed
    raw_df["selfemp"] = raw_df["wrkslf"].map({1: 1, 2: 0})
    raw_df["selfemp"] = raw_df["selfemp"].fillna(0)
    raw_df["selfemp"] = raw_df["selfemp"].astype("int")

    raw_df.drop("wrkslf", axis=1, inplace=True)

    # age: -> fill missing "age" cells with mean (4 missing)
    raw_df["age"].fillna(raw_df["age"].mean(), inplace=True)

    # sex: (gender) -> transform to binary "female" column
    raw_df["female"] = raw_df["sex"].map({2: 1, 1: 0})
    raw_df.drop("sex", axis=1, inplace=True)

    # educ: (Education in years) -> do nothing

    # relig: (Religion) -> do nothing

    # wrkgovt: (Gov employee or not) -> transform to binary govemp column
    raw_df["govemp"] = raw_df["wrkgovt"].map({1: 1, 2: 0})
    raw_df.drop("wrkgovt", axis=1, inplace=True)

    # prestg10: (prestige score of occupation) -> do nothing

    # indus10: (Industry of work) -> nominal, high cardinality, collapse to fewer with industry codes
    # reduces industries from 232 to 12
    raw_df["industry"] = raw_df["indus10"].map(ind_codes_labels)

    # map to categories
    raw_df["industry"] = raw_df["industry"].astype("category")
    raw_df.drop("indus10", axis=1, inplace=True)

    # marital: marital status -> do nothinh´g

    # partyid: (political party affiliation) -> do nothing

    # vote12: (if voted in 2012) -> transform to binary voted_12 column
    raw_df["voted_12"] = np.where(raw_df["vote12"] == 1, 1, 0)
    raw_df.drop("vote12", axis=1, inplace=True)

    # income16: (income category) -> ordinal categorical
    # TODO: check value counts and maybe collapse to fewer categories

    # nat variables: (opinion, if spending by govt is too little, about right or too much) -> do nothing

    # capppun: (favor or oppose death pen) -> transform to binary fav_death_pen column
    raw_df["fav_death_pen"] = raw_df["cappun"].map({1: 1, 2: 0})
    raw_df.drop("cappun", axis=1, inplace=True)

    # hispanic: (whether or not is hispanic) -> transform to binary column
    raw_df["hispanic"] = np.where(raw_df["hispanic"] == 1, 0, 1)

    # vetyears: (years in armed forces) -> do nothing

    # drop na rows
    raw_df.dropna(inplace=True)

    # convert some misspecified columns to int
    raw_df["govemp"] = raw_df["govemp"].astype("int")
    raw_df["fav_death_pen"] = raw_df["fav_death_pen"].astype("int")

    return raw_df


# analytical base table (abt)
abt = df.copy()

# run dataframe through preprocessing
abt = preprocessing(abt)

# show info
abt.info()

# print first 5 rows and dimensions of processed dataframe
print(f"Processed data - First rows: \n {abt.head()}")
print(f"Processed data - Dimensions: {abt.shape}")

# VISUALIZATION

# categorical variables
cat_vars = abt.select_dtypes(include=["category"]).columns

cat_cols = 3
cat_rows = math.ceil(len(cat_vars) / cat_cols)

fig, axes = plt.subplots(cat_rows, cat_cols, figsize=(14,8))
fig.suptitle("Categorical Variables", fontsize=12)

for index, key in enumerate(cat_vars):
    plt.subplot(cat_rows, cat_cols, index + 1)
    sns.countplot(data=abt, y=key)
fig.tight_layout()

# binary vars
bin_vars = abt.select_dtypes(include=["int"]).columns

bin_cols = 3
bin_rows = math.ceil(len(bin_vars) / bin_cols)

fig, axes = plt.subplots(bin_rows, bin_cols, figsize=(14,8))
fig.suptitle("Binary Variables", fontsize=12)

for index, key in enumerate(bin_vars):
    plt.subplot(bin_rows, bin_cols, index + 1)
    sns.countplot(data=abt, x=key)
fig.tight_layout()

# continuous vars
con_vars = abt.select_dtypes(include=["float"]).columns

con_cols = 3
con_rows = math.ceil(len(con_vars) / con_cols)

fig, axes = plt.subplots(con_rows, con_cols, figsize=(14,8))
fig.suptitle("Continuous Variables", fontsize=12)

for index, key in enumerate(con_vars):
    plt.subplot(con_rows, con_cols, index + 1)
    sns.histplot(data=abt, x=key)
fig.tight_layout()



## Models

# create target column
y = abt["trump"]

# create feature colummns
X = abt.drop(["trump"], axis=1)

# One Hot Encoding (create dummy vars for nominal categories), set drop_first=True to avoid dummy variable trap
X = pd.get_dummies(X, drop_first=True)

# TODO: Scale continuous variables
cont_vars = abt.select_dtypes(include=["float"]).columns
X[cont_vars] = StandardScaler().fit_transform(X[cont_vars])


# create train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1111
)


# feature selection: because of different data types lasso makes the most sense
def feat_selection(X_train, X_test, y_train):
    selector = SelectFromModel(estimator=LogisticRegression(C=1, penalty="l1", solver="saga", max_iter=10000, random_state=1234)).fit(X_train, y_train)
    sel_cols_idx = selector.get_support()
    sel_cols = X_train.iloc[:, sel_cols_idx].columns
    rem_cols = X_train.iloc[:, ~sel_cols_idx].columns
    print(f"Total features: {X_train.shape[1]}")
    print(f"Selected features: {len(sel_cols)}")
    print(f"Removed features: {len(rem_cols)}")

    X_train_sel = X_train[sel_cols]
    X_test_sel = X_test[sel_cols]
    
    return X_train_sel, X_test_sel

X_train_sel, X_test_sel = feat_selection(X_train, X_test, y_train)


## WITH HYPERPARAMETER TUNING
# Bagging: RandomForest, Boosting: AdaBoost and Gradient Boost, Stacking: StackingClassifier Ensemble of different models
# GaussianProcessClassifier, ADABoostClassifier,

# TODO: what algos need standardized/normalized features

# make pipelines
pipelines = {
    "LR": make_pipeline(
        LogisticRegression(penalty="none", max_iter=10000, random_state=1234)
    ),
    "LR_lasso": make_pipeline(
        LogisticRegression(
            penalty="l1", solver="saga", max_iter=10000, random_state=1234
        )
    ),
    "LR_ridge": make_pipeline(
        LogisticRegression(
            penalty="l2", solver="saga", max_iter=10000, random_state=1234
        )
    ),
    "LR_enet": make_pipeline(
        LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=10000, random_state=1234
        )
    ),
    "Ridge": make_pipeline(RidgeClassifierCV()),
    "DTC": make_pipeline(DecisionTreeClassifier(random_state=1234)),
    "RFC": make_pipeline(RandomForestClassifier()),
    "KN": make_pipeline(MinMaxScaler(), KNeighborsClassifier()),
    "GNB": make_pipeline(GaussianNB()),
    "LDA": make_pipeline(LinearDiscriminantAnalysis()),
    "SVC": make_pipeline(MinMaxScaler(), SVC(random_state=1234)),
    "GB": make_pipeline(GradientBoostingClassifier(random_state=1234)),
    "BDT": make_pipeline(BaggingClassifier(random_state=1234)),
    "NN": make_pipeline(MLPClassifier(random_state=1234, max_iter=1000)),
    "Stack": make_pipeline(StackingClassifier(estimators=[
        ('ridge', RidgeClassifierCV()),
        ('lda', LinearDiscriminantAnalysis())
        ], final_estimator=LogisticRegression(penalty="none", max_iter=10000, random_state=1234)))
}

# make hypergrid to find best parameters for each model
hypergrid = {
    "LR": {},
    "LR_lasso": {"logisticregression__C": np.logspace(-4, 4, 10)},
    "LR_ridge": {"logisticregression__C": np.logspace(-4, 4, 10)},
    "LR_enet": {
        "logisticregression__l1_ratio": np.arange(0, 1, 0.1),
        "logisticregression__C": np.logspace(-4, 4, 10),
    },
    "Ridge": {"ridgeclassifiercv__alphas": np.arange(0.1, 1, 0.1)},
    "DTC": {
        "decisiontreeclassifier__min_samples_split": [2, 4, 6],
        "decisiontreeclassifier__min_samples_leaf": [1, 2, 3],
    },
    "RFC": {
        "randomforestclassifier__n_estimators": [10, 100, 1000],
        "randomforestclassifier__max_features": ["sqrt", "log2"],
    },
    "KN": {
        "kneighborsclassifier__leaf_size": [10, 20, 30, 40, 50],
        "kneighborsclassifier__n_neighbors": np.arange(1, 20, 1),
        "kneighborsclassifier__p": [1, 2],
        "kneighborsclassifier__metric": ["euclidean", "manhattan", "minkowski"],
        "kneighborsclassifier__weights": ["uniform", "distance"],
    },
    "GNB": {"gaussiannb__var_smoothing": [1e-11, 1e-10, 1e-09]},
    "LDA": {},
    "SVC": {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": [1, 0.1, 0.01, 0.001],
        "svc__kernel": ["linear", "poly", "rbf", "sigmoid"],
    },
    "GB": {
        "gradientboostingclassifier__n_estimators": [10, 100, 1000],
        "gradientboostingclassifier__learning_rate": [0.001, 0.01, 0.1],
    },
    "BDT": {"baggingclassifier__n_estimators": [10, 100, 1000]},
    "NN": {
        "mlpclassifier__alpha": [0.0001, 0.05],
        "mlpclassifier__learning_rate": ['constant','adaptive']
    },
    "Stack": {}
}


# Hyperparameter Tuning (find best parameters for each model)
fit_models = {}
fit_models_sel = {}
print("Training models...")
for algo, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hypergrid[algo], cv=10, n_jobs=-1)
    model_sel = GridSearchCV(pipeline, hypergrid[algo], cv=10, n_jobs=-1)
    try:
        model.fit(X_train, y_train)  # without feature selection
        fit_models[algo] = model

        model_sel.fit(X_train_sel, y_train)  # with feature selection
        fit_models_sel[algo] = model_sel

        print(f"{algo} successful...")
    except NotFittedError as e:
        print(repr(e))
print("Training models done.")


# get best parameters
# fit_models["LR_lasso"].best_params_

# get all parameters with best parameters
# fit_models["LR"].best_estimator_.get_params()

        


# evaluate performance on test partition
results = {}
print("*****WITHOUT FEATURE SELECTION*****")
for algo, model in fit_models.items():
    print(f"Predicting {algo}...")

    y_hat = model.predict(X_test)

    classreport = classification_report(y_test, y_hat)
    conf_mat = confusion_matrix(y_test, y_hat)


    print(f"Accuracy Score: {accuracy_score(y_test, y_hat)}")
    print(f"Precision Score: {precision_score(y_test, y_hat)}")
    print(f"Recall Score: {recall_score(y_test, y_hat)}")
    print(f"Classification Report:\n {classreport}")
    print(f"Confusion Matrix:\n {conf_mat}")

    results[algo] = []
    results[algo].append(accuracy_score(y_test, y_hat))
    results[algo].append(precision_score(y_test, y_hat))
    results[algo].append(recall_score(y_test, y_hat))


results_df = pd.DataFrame.from_dict(results, orient='index', columns=["Accuracy", "Precision", "Recall"])

results_sel = {}
print("*****WITH FEATURE SELECTION*****")
for algo, model in fit_models_sel.items():
    print(f"Predicting {algo}...")

    y_hat = model.predict(X_test_sel)

    classreport = classification_report(y_test, y_hat)
    conf_mat = confusion_matrix(y_test, y_hat)


    print(f"Accuracy Score: {accuracy_score(y_test, y_hat)}")
    print(f"Precision Score: {precision_score(y_test, y_hat)}")
    print(f"Recall Score: {recall_score(y_test, y_hat)}")
    print(f"Classification Report:\n {classreport}")
    print(f"Confusion Matrix:\n {conf_mat}")

    results_sel[algo] = []
    results_sel[algo].append(accuracy_score(y_test, y_hat))
    results_sel[algo].append(precision_score(y_test, y_hat))
    results_sel[algo].append(recall_score(y_test, y_hat))



results_sel_df = pd.DataFrame.from_dict(results_sel, orient='index', columns=["Accuracy", "Precision", "Recall"])


x = "stop"