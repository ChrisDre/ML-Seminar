import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import math

# sklearn: processing, selection, analysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel

# sklearn: models:
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    StackingClassifier,
)


# set seed
np.random.seed(5000)

# DATA

# load variable list from excel sheet
df_vars = pd.read_excel("data/variables.xlsx", sheet_name="categories")

# get chosen variables
chosen_vars = df_vars["Variable"].dropna().unique().tolist()

# load gss survey data from 2018 (already pre-filtered for easier import)
# (use some with labels for easier processing)
df = pd.read_pickle("data/gss_2018_no_labels.pkl")
df_labels = pd.read_pickle("data/gss_2018_labels.pkl")

# filter for chosen trump variables
df = df[chosen_vars]
df_labels = df_labels[chosen_vars]

# replace some unlabeled variables with labeled variables
vars_with_labels = [
    "wrkstat",
    "relig",
    "marital",
    "natsoc",
    "natmass",
    "natpark",
    "natchld",
    "natsci",
    "natenrgy",
]
df[vars_with_labels] = df_labels[vars_with_labels]

# print first 5 rows and dimensions of unprocessed dataframe
print(f"\nUnprocessed data - First rows: \n {df.head()}")
print(f"\nUnprocessed data - Dimensions: \n {df.shape}")

# load NAICs to classify industries
ind_codes_xl = pd.read_excel("data/industry_codes.xlsx")
ind_codes_labels = dict(
    zip(ind_codes_xl["subsector"].to_list(), ind_codes_xl["short"].to_list())
)

# preprocessing function (clean up data and bring in correct format)
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
        raw_df["wrkstat"].isin(
            ["WORKING FULLTIME", "WORKING PARTTIME", "TEMP NOT WORKING"]
        ),
        0,
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

    # age: -> fill missing "age" cells with mean (>10 missing)
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

    # vote12: (if voted in 2012) -> transform to binary voted_12 column
    raw_df["voted_12"] = np.where(raw_df["vote12"] == 1, 1, 0)
    raw_df.drop("vote12", axis=1, inplace=True)

    # income16: (income category) ordinal categorical -> do nothing

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

    # capitalize all categorical variables
    raw_df = raw_df.apply(
        lambda x: x.str.capitalize().astype("category")
        if (x.dtype == "category")
        else x
    )

    return raw_df


# analytical base table (abt)
abt = df.copy()

# run abt through preprocessing function
abt = preprocessing(abt)

# show info
abt.info()

# print first 5 rows and dimensions of processed dataframe
print(f"\nProcessed data - First rows: \n{abt.head()}")
print(f"\nProcessed data - Dimensions: \n{abt.shape}")


# VISUALIZATION

# categorical variables (nat variables separately)
cat_vars = abt[["wrkstat", "relig", "marital", "industry"]]
cat_vars = cat_vars.rename(
    columns={
        "wrkstat": "Work Status",
        "relig": "Religion",
        "marital": "Marital Status",
        "industry": "Industry",
    }
)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for index, key in enumerate(cat_vars):
    plt.subplot(2, 2, index + 1)
    sns.countplot(
        data=cat_vars,
        x=key,
        order=cat_vars[key].value_counts().index,
        palette="colorblind",
    )
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(key, fontsize=15)
# fig.delaxes(axes[1][2])
fig.tight_layout()
plt.savefig("plots/categories.png", dpi=300)
plt.close(fig)


# nat variables (spending as a country -> government)
spend_vars = abt[["natsoc", "natmass", "natpark", "natchld", "natsci", "natenrgy"]]
spend_vars = spend_vars.rename(
    columns={
        "natsoc": "Social Security",
        "natmass": "Mass Transportation",
        "natpark": "Parks and Recreation",
        "natchld": "Childcare (Assistance)",
        "natsci": "Scientific Research",
        "natenrgy": "Alternative Energy",
    }
)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for index, key in enumerate(spend_vars):
    plt.subplot(2, 3, index + 1)
    sns.countplot(
        data=spend_vars,
        x=key,
        order=["Too little", "About right", "Too much"],
        palette="colorblind",
    )
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(key, fontsize=16)
fig.tight_layout()
plt.savefig("plots/categories_spending.png", dpi=300)
plt.close(fig)


# binary vars
bin_vars = abt[
    ["hispanic", "trump", "selfemp", "female", "govemp", "voted_12", "fav_death_pen"]
]
bin_vars = bin_vars.rename(
    columns={
        "hispanic": "Hispanic",
        "trump": "Voted Trump",
        "selfemp": "Self Employed",
        "female": "Female",
        "govemp": "Government Employee",
        "voted_12": "Voted in 2012",
        "fav_death_pen": "Favor Death Penalty (Murder)",
    }
)

fig, axes = plt.subplots(3, 3, figsize=(14, 8))
for index, key in enumerate(bin_vars):
    plt.subplot(3, 3, index + 1)
    sns.countplot(data=bin_vars, x=key, palette="colorblind")
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(key, fontsize=16)
fig.delaxes(axes[2][1])
fig.delaxes(axes[2][2])
fig.tight_layout()
plt.savefig("plots/binary.png", dpi=300)
plt.close(fig)


# continuous vars
con_vars = abt[["age", "educ", "prestg10", "income16", "vetyears", "hours_worked"]]
con_vars = con_vars.rename(
    columns={
        "age": "Age",
        "educ": "Education in Years",
        "prestg10": "Prestige Score of Occupation",
        "income16": "Income Category",
        "vetyears": "Time Category Military",
        "hours_worked": "Work Hours per Week",
    }
)

fig, axes = plt.subplots(3, 2, figsize=(14, 8))
for index, key in enumerate(con_vars):
    plt.subplot(3, 2, index + 1)
    sns.histplot(data=con_vars, x=key, discrete=True, color="#56b4e9")
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(key, fontsize=16)
fig.tight_layout()
plt.savefig("plots/cont.png", dpi=300)
plt.close(fig)


## MODELS

# create target variable
y = abt["trump"]

# create feature variables
X = abt.drop(["trump"], axis=1)

# One Hot Encoding (create dummy vars for nominal categories), set drop_first=True to avoid dummy variable trap
X = pd.get_dummies(X, drop_first=True)

# scale (standardize) continuous variables
cont_vars = abt.select_dtypes(include=["float"]).columns
X[cont_vars] = StandardScaler().fit_transform(X[cont_vars])

# create train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=99
)

# feature selection: because of different data types Lasso makes the most sense
def feat_selection(X_train, X_test, y_train):
    selector = SelectFromModel(
        estimator=LogisticRegression(
            C=1, penalty="l1", solver="saga", max_iter=10000, random_state=1234
        )
    ).fit(X_train, y_train)
    sel_cols_idx = selector.get_support()
    sel_cols = X_train.iloc[:, sel_cols_idx].columns
    rem_cols = X_train.iloc[:, ~sel_cols_idx].columns
    print(f"Total features: {X_train.shape[1]}")
    print(f"Selected features: {len(sel_cols)}")
    print(f"Removed features: {len(rem_cols)}")

    X_train_fs = X_train[sel_cols]
    X_test_fs = X_test[sel_cols]

    return X_train_fs, X_test_fs


# create new data sets with selected features
X_train_fs, X_test_fs = feat_selection(X_train, X_test, y_train)

# init all models in pipeline (can be used to combine functions)
pipelines = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(penalty="none", max_iter=10000, random_state=1234)
    ),
    "Lasso": make_pipeline(
        LogisticRegression(
            penalty="l1", solver="saga", max_iter=10000, random_state=1234
        )
    ),
    "Ridge": make_pipeline(
        LogisticRegression(
            penalty="l2", solver="saga", max_iter=10000, random_state=1234
        )
    ),
    "Elastic Net": make_pipeline(
        LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=10000, random_state=1234
        )
    ),
    "Ridge Classifier": make_pipeline(RidgeClassifier(random_state=1234)),
    "Decision Tree": make_pipeline(DecisionTreeClassifier(random_state=1234)),
    "Random Forest": make_pipeline(RandomForestClassifier(random_state=1234)),
    "Bagged Trees": make_pipeline(BaggingClassifier(random_state=1234)),
    "Boosted Trees": make_pipeline(GradientBoostingClassifier(random_state=1234)),
    "k-neighbors": make_pipeline(MinMaxScaler(), KNeighborsClassifier()),
    "Neural Network": make_pipeline(
        MLPClassifier(solver="lbfgs", random_state=1234, max_iter=1000)
    ),
    "Ensemble_1": make_pipeline(
        StackingClassifier(
            estimators=[
                ("model1", RidgeClassifier(random_state=1234)),
                ("model2", RandomForestClassifier(random_state=1234)),
            ],
            final_estimator=LogisticRegression(
                penalty="l1", solver="saga", max_iter=10000, random_state=1234
            ),
        )
    ),
}

# Init Hyperparameter Tuning (specify model parameters to loop through and find best)
hypergrid = {
    "Logistic Regression": {"logisticregression__class_weight": [None, "balanced"]},
    "Lasso": {
        "logisticregression__C": np.logspace(-4, 4, 10),
        "logisticregression__class_weight": [None, "balanced"],
    },
    "Ridge": {
        "logisticregression__C": np.logspace(-4, 4, 10),
        "logisticregression__class_weight": [None, "balanced"],
    },
    "Elastic Net": {
        "logisticregression__l1_ratio": np.arange(0, 1, 0.1),
        "logisticregression__C": np.logspace(-4, 4, 10),
        "logisticregression__class_weight": [None, "balanced"],
    },
    "Ridge Classifier": {
        "ridgeclassifier__alpha": np.arange(0.1, 1, 0.1),
        "ridgeclassifier__class_weight": [None, "balanced"],
    },
    "Decision Tree": {
        "decisiontreeclassifier__min_samples_split": [2, 3, 4],
        "decisiontreeclassifier__min_samples_leaf": [20, 30, 40],
        "decisiontreeclassifier__criterion": ["gini", "entropy", "log_loss"],
        "decisiontreeclassifier__max_depth": [None, 8, 16],
        "decisiontreeclassifier__max_features": [None, "sqrt", "log2"],
    },
    "Random Forest": {
        "randomforestclassifier__n_estimators": [30, 40, 50, 60],
        "randomforestclassifier__max_depth": [None, 6, 8, 10, 12],
        "randomforestclassifier__min_samples_split": [2, 4],
        "randomforestclassifier__min_samples_leaf": [2, 4, 6],
        "randomforestclassifier__max_features": ["sqrt", "log2", None],
        "randomforestclassifier__class_weight": [None, "balanced"],
    },
    "Bagged Trees": {
        "baggingclassifier__n_estimators": [20, 30, 40, 50],
        "baggingclassifier__max_samples": np.arange(0.1, 1, 0.1),
        "baggingclassifier__max_features": np.arange(0.1, 1, 0.1),
    },
    "Boosted Trees": {
        "gradientboostingclassifier__n_estimators": [100, 200, 300],
        "gradientboostingclassifier__learning_rate": [0.005, 0.01, 0.02],
        "gradientboostingclassifier__loss": ["log_loss", "exponential"],
        "gradientboostingclassifier__subsample": np.arange(0.1, 1, 0.1),
        "gradientboostingclassifier__criterion": ["friedman_mse", "squared_error"],
    },
    "k-neighbors": {
        "kneighborsclassifier__n_neighbors": np.arange(1, 20, 1),
        "kneighborsclassifier__leaf_size": [2, 3, 4],
        "kneighborsclassifier__p": [1, 2],
        "kneighborsclassifier__metric": ["euclidean", "manhattan", "minkowski"],
        "kneighborsclassifier__weights": ["uniform", "distance"],
    },
    "Neural Network": {
        "mlpclassifier__hidden_layer_sizes": [(75,), (100,), (75, 75), (100, 100)],
        "mlpclassifier__alpha": [0.0001, 0.00015, 0.0002],
    },
    "Ensemble_1": {
        "stackingclassifier__model1__alpha": np.arange(0.1, 1, 0.2),
        "stackingclassifier__model1__class_weight": ["balanced"],
        "stackingclassifier__model2__class_weight": ["balanced"],
        "stackingclassifier__model2__n_estimators": [30, 50],
        "stackingclassifier__final_estimator__C": np.logspace(-3, 3, 5),
        "stackingclassifier__final_estimator__class_weight": ["balanced"],
    },
}

# Train Hyperparameter Tuning (find best parameters for each model through 10-fold CV)
fit_models = {}  # with hyperparameter tuning but without feature selection
fit_models_fs = {}  # with hyperparameter tuning and feature selection
print("\n***** TRAINING MODELS *****")
for algo, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hypergrid[algo], cv=10, n_jobs=-1)
    model_fs = GridSearchCV(pipeline, hypergrid[algo], cv=10, n_jobs=-1)
    try:
        model.fit(X_train, y_train)  # tuning without feature selection
        fit_models[algo] = model

        model_fs.fit(X_train_fs, y_train)  # tuning with with feature selection
        fit_models_fs[algo] = model_fs

        print(f"{algo} successful.")
    except NotFittedError as e:
        print(repr(e))
print("***** TRAINING MODELS DONE *****\n")


# if you want so see the selected hyperparameters of the model:
# fit_models["LR_lasso"].best_params_

# if you want to see all parameters of the model:
# fit_models["LR"].best_estimator_.get_params()


## evaluate performance on test partition

# WITH HYPERPARAMETER TUNING AND NO FEATURE SELECTION
results = {}
print("***** RESULTS WITH TUNING BUT WITHOUT FEATURE SELECTION *****\n")
for algo, model in fit_models.items():
    print(f"Predicting {algo}...")

    y_hat = model.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_hat)

    # print(f"Accuracy Score: {accuracy_score(y_test, y_hat)}")
    # print(f"Precision Score: {precision_score(y_test, y_hat)}")
    # print(f"Recall Score: {recall_score(y_test, y_hat)}")
    # print(f"Confusion Matrix:\n {conf_mat}\n")

    results[algo] = []
    results[algo].append(accuracy_score(y_test, y_hat))
    results[algo].append(precision_score(y_test, y_hat))
    results[algo].append(recall_score(y_test, y_hat))
    results[algo].append(conf_mat)


results_df = pd.DataFrame.from_dict(
    results,
    orient="index",
    columns=["Accuracy", "Precision", "Recall", "Confusion Matrix"],
)

# WITH HYPERPARAMETER TUNING AND FEATURE SELECTION
results_fs = {}
print("***** RESULTS WITH TUNING AND FEATURE SELECTION *****")
for algo, model in fit_models_fs.items():
    print(f"Predicting {algo}...")

    y_hat = model.predict(X_test_fs)

    conf_mat = confusion_matrix(y_test, y_hat)

    # print(f"Accuracy Score: {accuracy_score(y_test, y_hat)}")
    # print(f"Precision Score: {precision_score(y_test, y_hat)}")
    # print(f"Recall Score: {recall_score(y_test, y_hat)}")
    # print(f"Confusion Matrix:\n {conf_mat}\n")

    results_fs[algo] = []
    results_fs[algo].append(accuracy_score(y_test, y_hat))
    results_fs[algo].append(precision_score(y_test, y_hat))
    results_fs[algo].append(recall_score(y_test, y_hat))
    results_fs[algo].append(conf_mat)


results_fs_df = pd.DataFrame.from_dict(
    results_fs,
    orient="index",
    columns=["Accuracy", "Precision", "Recall", "Confusion Matrix"],
)

# save results as pickle
# without feature selection
with open("models/model_results.pkl", "wb") as handle:
    pickle.dump(fit_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with feature selection
with open("models/model_results_fs.pkl", "wb") as handle:
    pickle.dump(fit_models_fs, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Visualize results
# create combined "accuracy" dataframe
all_models_accuracy = pd.concat(
    [results_df["Accuracy"], results_fs_df["Accuracy"]],
    axis=1,
    keys=["No feature selection", "Feature Selection"],
)
all_models_accuracy.sort_values(
    by=["No feature selection"], ascending=False, inplace=True
)
all_models_accuracy["Model"] = all_models_accuracy.index
all_models_accuracy = all_models_accuracy.melt(id_vars="Model", value_name="Accuracy")

# plot results
fig, ax = plt.subplots(figsize=(14, 8))
sns.barplot(
    x="Model",
    y="Accuracy",
    hue="variable",
    data=all_models_accuracy,
    palette=["#55aa99", "#cde4df"],
)
plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
plt.ylim([0.5, 0.8])
plt.title("Accuracy", fontsize=15)
ax.legend(ncol=2, loc="upper right", frameon=True)
ax.set(ylabel="", xlabel="")
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.2f")
fig.tight_layout()
plt.savefig("plots/results.png", dpi=300)
plt.close(fig)


# visualize trees
trees = {
    "decision_tree": fit_models["Decision Tree"].best_estimator_.named_steps[
        "decisiontreeclassifier"
    ],
    "random_forest_tree": fit_models["Random Forest"]
    .best_estimator_.named_steps["randomforestclassifier"]
    .estimators_[0],
    "bagged_tree": fit_models["Bagged Trees"]
    .best_estimator_.named_steps["baggingclassifier"]
    .estimators_[0],
    "boosted_tree": fit_models["Boosted Trees"]
    .best_estimator_.named_steps["gradientboostingclassifier"]
    .estimators_[0][0],
}

for name, tree in trees.items():
    fig = plt.figure(figsize=(14, 8))
    _ = plot_tree(
        tree,
        feature_names=X_train.columns,
        filled=True,
        class_names=["Voted Other", "Voted Trump"],
    )
    plt.savefig("plots/" + name + ".png", dpi=300)
    plt.close(fig)

# feature importance for decision tree
dtc_feature_importance = pd.DataFrame(
    trees["decision_tree"].feature_importances_, index=X_train.columns
).sort_values(by=0, ascending=False)

# TODO: better plot
fig, ax = plt.subplots(figsize=(14, 8))
sns.barplot(
    x=dtc_feature_importance.index[0:10],
    y=dtc_feature_importance[0][0:10],
)
plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
fig.tight_layout()
plt.savefig("plots/decision_tree_features.png", dpi=300)
plt.close(fig)

x = "stop"
