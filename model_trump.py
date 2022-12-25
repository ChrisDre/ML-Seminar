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
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import linear_model, tree
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError


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
def preprocessing(raw_df):

    # reset index
    raw_df.reset_index(drop=True, inplace=True)

    # year: -> drop "year" column
    raw_df.drop("year", axis=1, inplace=True)

    # id: -> do nothing

    # pres16: (Target Variable, 2016 presidential vote) -> remove rows with missing vote in 2016 (1 = Clinton, 2 = Trump, 3 = other candidate)
    raw_df = raw_df[raw_df.pres16.isin([1, 2, 3])].copy()

    # pres16: transform to binary
    raw_df["trump"] = np.where(raw_df["pres16"] == 2, 1, 0)
    raw_df.drop("pres16", axis=1, inplace=True)

    # wrkstat: (Work Status) -> nominal categorical
    raw_df["wrkstat"] = raw_df["wrkstat"].astype("category")

    # hrs1: (Hours worked last week) -> replace missing values in hrs1 with hrs2 or zero for retired/unemp etc.
    raw_df["hours_worked"] = raw_df["hrs1"].where(~raw_df["wrkstat"].isin([4, 5, 6, 7, 8]), 0)
    raw_df["hours_worked"] = np.where(raw_df["wrkstat"] == 3, raw_df["hrs2"], raw_df["hours_worked"])
    raw_df.dropna(subset=["hours_worked"], inplace=True)
    raw_df.drop("hrs1", axis=1, inplace=True)
    raw_df.drop("hrs2", axis=1, inplace=True)

    # wrkslf: (Self employed or not) -> if missing but retired/unemp etc. set to not self employed
    raw_df["selfemp"] = raw_df["wrkslf"].map({1: 1, 2: 0})
    raw_df["selfemp"] = raw_df["selfemp"].fillna(0)
    raw_df["selfemp"] = raw_df["selfemp"].astype('int')


    raw_df.drop("wrkslf", axis=1, inplace=True)

    # age: -> fill missing "age" cells with mean (4 missing)
    raw_df["age"].fillna(raw_df["age"].mean(), inplace=True)

    # sex: (gender) -> transform to binary "female" column
    raw_df["female"] = raw_df["sex"].map({2: 1, 1: 0})
    raw_df.drop("sex", axis=1, inplace=True)

    # educ: (Education in years) -> do nothing

    # relig: (Religion) -> nominal categorical
    raw_df["relig"] = raw_df["relig"].astype("category")


    # wrkgovt: (Gov employee or not) -> transform to binary govemp column
    raw_df["govemp"] = raw_df["wrkgovt"].map({1: 1, 2: 0})
    raw_df.drop("wrkgovt", axis=1, inplace=True)

    # prestg10: (prestige score of occupation) -> do nothing

    # indus10: (Industry of work) -> nominal, high cardinality, collapse to fewer with industry codes
    # reduces industries from 232 to 12
    raw_df["industry"] = raw_df["indus10"].map(ind_codes)
    raw_df["industry"] = raw_df["industry"].astype("category")
    raw_df.drop("indus10", axis=1, inplace=True)

    # marital: (marriage status) -> nominal categorical
    raw_df["marital"] = raw_df["marital"].astype("category")

    # partyid: (political party affiliation) -> nominal categorical
    raw_df["partyid"] = raw_df["partyid"].astype("category")

    # vote12: (if voted in 2012) -> transform to binary voted_12 column
    raw_df["voted_12"] = np.where(raw_df["vote12"] == 1, 1, 0)
    raw_df.drop("vote12", axis=1, inplace=True)

    # vote16: (same as vote12)
    raw_df["voted_16"] = np.where(raw_df["vote16"] == 1, 1, 0)
    raw_df.drop("vote16", axis=1, inplace=True)

    # income16: (income category) -> ordinal categorical
    # TODO: check value counts and maybe collapse to fewer categories

    # nat variables: (opinion, if spending by govt is too little, about right or too much) -> nominal categorical
    for col in ["natsoc", "natmass", "natpark", "natchld", "natsci", "natenrgy"]:
        raw_df[col] = raw_df[col].astype("category")

    # capppun: (favor or oppose death pen) -> transform to binary fav_death_pen column
    raw_df["fav_death_pen"] = raw_df["cappun"].map({1: 1, 2: 0})
    raw_df.drop("cappun", axis=1, inplace=True)

    # hispanic: (whether or not is hispanic) -> transform to binary column
    raw_df["hispanic"] = np.where(raw_df["hispanic"] == 1, 0, 1)

    # vetyears: (years in armed forces) -> do nothing

    # drop na rows
    raw_df.dropna(inplace=True)

    # convert some misspecified columns to int
    raw_df["govemp"] = raw_df["govemp"].astype('int')
    raw_df["fav_death_pen"] = raw_df["fav_death_pen"].astype('int')


    return raw_df


# analytical base table (abt)
abt = df.copy()

# run dataframe through preprocessing
abt = preprocessing(abt)

# show info
abt.info()

# TODO: visualization and summarization
# TODO: feature selection
# TODO: find classification algos
# Models: Logitstic Regression, Naive Bayes Classifier, K-Nearest Neighbor, Decision Tree, Random Forest, Artificial Neural Networks, Support Vector Machines:
# Evaluation: CV, Classification Report, ROC Curve, 


## Models

# create target column
y = abt["trump"]

# create feature colummns
X = abt.drop(["trump", "id"], axis=1)

# One Hot Encoding (create dummy vars for nominal categories), set drop_first=True to avoid dummy variable trap
X = pd.get_dummies(X, drop_first=True)

# remove relig_8 to have same df as in R, (relig_8 does not exist in our processed data anyway)
X.drop("relig_8.0", axis=1, inplace=True)


# TODO: Scale continuous variables
cont_vars = abt.select_dtypes(include=['float']).columns
X[cont_vars] = StandardScaler().fit_transform(X[cont_vars])


# create train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1111
)


# set seed
np.random.seed(5000)


## WITH HYPERPARAMETER TUNING
# make pipelines
pipelines = {
    "LR": make_pipeline(LogisticRegression(solver="saga", max_iter=10000, random_state=1234)),
    "LR_enet": make_pipeline(LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000, random_state=1234)),
    "Ridge": make_pipeline(RidgeClassifierCV()),
    "DTC": make_pipeline(DecisionTreeClassifier(random_state=1234)),
    "RFC": make_pipeline(RandomForestClassifier()),
    "KN": make_pipeline(KNeighborsClassifier()),
    "GNB": make_pipeline(GaussianNB()),
    "LDA": make_pipeline(LinearDiscriminantAnalysis()),
    "SVC": make_pipeline(SVC(random_state=1234)),
    "GB": make_pipeline(GradientBoostingClassifier(random_state=1234))
}

# make hypergrid to find best parameters for each model
hypergrid = {
    "LR": {
        "logisticregression__penalty": ["none", "l1", "l2"]
        },
    "LR_enet": {
        "logisticregression__l1_ratio": [0.1, 0.5, 0.9]
        },
    "Ridge": {
        "ridgeclassifiercv__alphas": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99]
        },
    "DTC": {
        "decisiontreeclassifier__min_samples_split": [2, 4, 6],
        "decisiontreeclassifier__min_samples_leaf": [1, 2, 3]
        },
    "RFC": {
        "randomforestclassifier__n_estimators": [50, 100, 200],
        },
    "KN": {
        "kneighborsclassifier__leaf_size": [10, 20, 30, 40, 50],
        "kneighborsclassifier__n_neighbors": [2, 3, 5, 7, 10],
        "kneighborsclassifier__p": [1, 2],
        },
    "GNB": {
        "gaussiannb__var_smoothing": [1e-11, 1e-10, 1e-09]
        },
    "LDA": {
        },
    "SVC": {"svc__C": [0.1,1, 10, 100],
            "svc__gamma": [1,0.1,0.01,0.001],
            "svc__kernel": ['rbf', 'poly', 'sigmoid']
        },
    "GB": {"gradientboostingclassifier__n_estimators": [100, 200, 300]

    }
}




fit_models = {}

for algo, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hypergrid[algo], cv=10, n_jobs=-1)
    print("Training models...")
    try:
        model.fit(X_train, y_train)
        fit_models[algo] = model
        print(f"{algo} successful...")
    except NotFittedError as e:
        print(repr(e))
    print("Training models done.")




# evaluate performance on test partition
for algo, model in fit_models.items():
    print(f"Predicting {algo}...")
    y_hat = model.predict(X_test)
    print(f"Accuracy Score: {accuracy_score(y_test, y_hat)}")
    print(f"Precision Score: {precision_score(y_test, y_hat)}")
    print(f"Recall Score: {recall_score(y_test, y_hat)}")
    







## STANDARD MODEL WITHOUT HYPERPARAMETER TUNING
# init list for comparison
models = []


def algo_models(X_train, X_test, y_train, y_test):

    models.append(("LR", LogisticRegression(penalty='none', solver="saga", max_iter=10000)))
    models.append(
        ("LR_L1", LogisticRegression(penalty="l1", solver="saga", max_iter=10000))
    )
    models.append(
        ("LR_L2", LogisticRegression(penalty="l2", solver="saga", max_iter=10000))
    )
    models.append(
        (
            "LR_EN",
            LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=10000
            ),
        )
    )
    models.append(("Ridge", RidgeClassifierCV()))
    models.append(("CART", DecisionTreeClassifier()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("NB", GaussianNB()))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append(("SVM", SVC()))

    outcomes = []
    names = []

    # loop through models
    print(f"********** CV **********")
    for name, model in models:
        # 5-Fold Cross Validation (train, validate)
        v_results = cross_val_score(
            model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
        )
        print(f"5-CV Accuracy for {name}: {v_results.mean()}")
        outcomes.append(v_results)
        names.append(name)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticklabels(names)
    plt.boxplot(outcomes)


algo_models(X_train, X_test, y_train, y_test)


# Evaluating and predicting models
for name, model in models:
    trained_model = model.fit(X_train, y_train)

    # prediction
    y_pred = trained_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    classreport = classification_report(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    coeffs = pd.DataFrame(zip(X.columns, trained_model.coef_[0]))


    print(f"******************** {name} ********************")
    print(f"Accuracy: {acc}")
    print(f"Classification Report:\n {classreport}")
    print(f"Confusion Matrix:\n {conf_mat}")
    print(f"Coefficients: {coeffs.to_string()}")





x = "Stop"