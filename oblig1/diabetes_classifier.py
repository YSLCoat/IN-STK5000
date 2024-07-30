import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.neural_network
from sklearn import metrics
from collections import Counter as counter
from argparse import ArgumentParser
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2        # Categorical vs categorical
from sklearn.feature_selection import f_classif   # Numeric vs categorical
from scipy.stats import chi2_contingency

# TASK 1

parser = ArgumentParser()
parser.add_argument("-f", "--file", help="path to csv file")
args = parser.parse_args()
csv_path = args.file

diabetes_df = pd.read_csv(csv_path)
diabetes_df.describe(include="object")

categorical_cols = diabetes_df.describe(include='object').columns
for i in categorical_cols:
    diabetes_df[i] = diabetes_df[i].str.lower()
    
# List those with more than 1 missing value
na_rows = list(diabetes_df.loc[diabetes_df.isnull().sum(1)>1].index)
print("index of individuals with 2 or more nan-values: ", na_rows)

print("\n")
print("number of individuals before dropping those with 2 or more nan-values: ",len(diabetes_df))
# Drop these rows, and confirm we now have 520 - 13 = 507 rows
diabetes_df = diabetes_df.drop(na_rows)
print("number of individuals after dropping those with 2 or more nan-values: ",len(diabetes_df))

# TASK 2 

empty_row = ["", "", ""]
empty_index = " "

columns = ["Mean(SD)", "Median", "Min to Max"]
index = ["\033" + "Continuous:", empty_index]
data = [empty_row, empty_row]

for i in diabetes_df.describe(include = "float64"):
    data.append([str(float("{:.2f}".format(diabetes_df[i].mean()))) + " (" + str(float("{:.2f}".format(diabetes_df[i].std())))
                 + ")", str(float("{:.2f}".format(diabetes_df[i].median()))),
                 str(float("{:.2f}".format(diabetes_df[i].min()))) + " - " + str(diabetes_df[i].max())])
    index.append(i)
    
new_frame_float64 = pd.DataFrame(data, index, columns)

empty_row = ["", "", ""]
empty_index = " "

columns = ["Mean(SD)", "Median", "Min to Max"]
index = [empty_index, empty_index, "Categorical:", empty_index, empty_index]
data = [empty_row, empty_row, ["\033" + "Undergroup", "\033" + "N", ""], empty_row, empty_row]


for i in diabetes_df.describe(include = "object"):
    list_count = counter(diabetes_df[i])
    count = 0

    for j in list_count:
        if count > 0:
            index.append(" ")
        else:
            index.append(i)
            count += 1

        data.append([j, list_count[j], " "])

new_frame_object = pd.DataFrame(data, index, columns)

frames = [new_frame_float64, new_frame_object]

boxplot = diabetes_df.boxplot(column=["Age", "Urination", "Weight", "Height"])
plt.show()

boxplot = diabetes_df.boxplot(column=["Urination"])
plt.show()

boxplot = diabetes_df.boxplot(column=["Age"])
plt.show()

boxplot = diabetes_df.boxplot(column=["Height"])
plt.show()



# We want to keep these
print(diabetes_df[(diabetes_df['Age'] <= 110) & (diabetes_df['Age'] >= 0) | diabetes_df['Age'].isnull()])
# We want to discard these
print(diabetes_df[(diabetes_df['Age'] > 110) | (diabetes_df['Age'] < 0)])

print("number of individuals before dropping those with age mistakes: ",len(diabetes_df))
diabetes_df = diabetes_df[(diabetes_df['Age'] <= 110) & (diabetes_df['Age'] >= 0) | diabetes_df['Age'].isnull()]
print("number of individuals after dropping those with age mistakes: ",len(diabetes_df))


# Check that there are no strange values
print("Strange values (from 2 to 130): ", diabetes_df["Height"][(diabetes_df["Height"]<130) & (diabetes_df["Height"]>2)].count())

# Scale all the values given in meters:
diabetes_df.loc[diabetes_df['Height'] < 2, "Height"] = diabetes_df['Height'] * 100

# And plot to confirm
hist = diabetes_df.hist(column=["Height"], bins = 30)
plt.show()


# list outliers
diabetes_df["Urination"][diabetes_df["Urination"]>4]


print("number of individuals before dropping those with urination above 4: ", len(diabetes_df))
# remove all outliers
diabetes_df = diabetes_df[(diabetes_df['Urination'] < 4) | diabetes_df['Urination'].isnull()]
print("number of individuals after dropping those with urination above 4: ",len(diabetes_df))

# and plot to confirm
# noinspection PyRedeclaration
hist = diabetes_df.hist(column=["Urination"], bins = 30)
plt.show()

diabetes_df_high_urination = diabetes_df[diabetes_df["Urination"]>2.45]
high_uri_diabetes_pos = len(diabetes_df_high_urination[(diabetes_df_high_urination["Diabetes"] == "positive")].value_counts(normalize=True))
high_uri = len(diabetes_df_high_urination)
print("Ratio, high urination: %d / %d = %.3f" % (high_uri_diabetes_pos, high_uri, (high_uri_diabetes_pos / high_uri)))

diabetes_df_low_urination = diabetes_df[diabetes_df["Urination"]<=2.45]
low_uri_diabetes_pos = len(diabetes_df_low_urination[(diabetes_df_low_urination["Diabetes"] == "positive")].value_counts(normalize=True))
low_uri = len(diabetes_df_low_urination)
print("Ratio, low urination: %d / %d = %.3f" % (low_uri_diabetes_pos, low_uri, (low_uri_diabetes_pos / low_uri)))

# Plot histogram of weight distribution
# noinspection PyRedeclaration
hist = diabetes_df.hist(column=["Weight"], figsize=(20,10), bins = 100)
plt.show()

diabetes_df.plot.scatter("Weight", 'Height', c = ['r' if t == "positive" else 'b' for t in diabetes_df.Diabetes])
plt.show()

diabetes_df["BMI"] = diabetes_df["Weight"] / ((diabetes_df["Height"]/100)**2)
#diabetes_df["BMI"]
# noinspection PyRedeclaration
hist = diabetes_df.hist(column=["BMI"], figsize=(20,10), bins = 100)


print("number of individuals before dropping those with BMI below or equal to 15: ", len(diabetes_df))
diabetes_df = diabetes_df[(diabetes_df['BMI'] >= 15) | diabetes_df['BMI'].isnull()]
print("number of individuals after dropping those with BMI below or equal to 15: ", len(diabetes_df))
diabetes_df.describe()

print("number of individuals that are not white = ", len(diabetes_df[(diabetes_df["Race"] != "white")]))
print("number of individuals before dropping non-white rows: ", len(diabetes_df))
diabetes_df = diabetes_df[(diabetes_df["Race"] == "white")]
diabetes_df.drop(columns=["Race"], axis=1, inplace=True)
print("number of individuals after dropping non-white rows: ", len(diabetes_df))
diabetes_df.describe(include="all")

diabetes_df.plot.scatter("Weight", 'Height', c = ['r' if t == "yes" else 'b' for t in diabetes_df.Obesity])
plt.show()

# noinspection PyRedeclaration
hist = diabetes_df.hist(column=["BMI"], figsize=(20,10), bins = 100)
one_hot = pd.get_dummies(diabetes_df["Obesity"], drop_first=True)
diabetes_df = pd.concat([diabetes_df, one_hot], axis=1)
diabetes_df.rename(columns={'yes': 'Obesity_yes'}, inplace=True)

diabetes_df.groupby(pd.cut(diabetes_df['BMI'],[0,27, 28, 29, 30, 31,32]))["Obesity_yes"].mean().plot.bar()
plt.ylim(0,2)
plt.show()
diabetes_df.describe()
#diabetes_df.groupby([pd.cut(diabetes_df['BMI'], [0,27, 28, 29,40]), "Obesity_yes"]).size().reset_index(name='count')

counts = counter(diabetes_df["Obesity"].isna())
print(counts)
print("number of individuals with nan in obesity: ", counts[True])
print(diabetes_df[diabetes_df['Obesity'].isna() == True])
diabetes_df.Obesity.describe()

# Fill inn missing values
diabetes_df['Obesity'] = np.where((diabetes_df["BMI"] >= 30) & (diabetes_df['Obesity'].isna() == True), "yes", diabetes_df['Obesity'])
diabetes_df['Obesity'] = np.where((diabetes_df["BMI"] < 30) & (diabetes_df['Obesity'].isna() == True), "no", diabetes_df['Obesity'])
diabetes_df.drop(columns=["Obesity_yes"], axis=1, inplace=True)
diabetes_df["Obesity"].describe()

print("number of individuals before dropping rows with nan-values in the categories: ",len(diabetes_df))

diabetes_df.dropna(subset = ["Age", "Gender", "Height", "Weight", "GP", "Occupation"], inplace=True)

print("number of individuals after dropping rows with nan-values in the categories: ", len(diabetes_df))

def fill_na_df(a):
      # at this point, all other columns are dealt with
  nan_columns = a.columns[a.isnull().any()]
  for col in nan_columns:
    # rate of yes vs no's
    value_ratios = a[col].value_counts(normalize=True)
    missing = a[col].isnull() # rows with a missing value
    # pick randomly, weighted by ratio
    a.loc[missing,col] = np.random.choice(value_ratios.index, size=len(a[missing]),p=value_ratios.values)

  return a

diabetes_df = fill_na_df(diabetes_df)
diabetes_df.info()

not_onehot_cols = ['Age', 'Height', 'Weight', 'Temperature', 'Urination', 'GP', 'Occupation']
# noinspection SpellCheckingInspection
categoricals = diabetes_df.drop(not_onehot_cols, axis=1)
one_hot = pd.get_dummies(categoricals, drop_first=True)
one_hot.head(5)

not_onehot_df = diabetes_df[not_onehot_cols]
not_onehot_df.describe(include="all")
diabetes_df = pd.concat([not_onehot_df, one_hot], axis=1)
diabetes_df.info()

#df_urination_high = diabetes_df[:]
df_urination_high = diabetes_df.copy()
df_urination_high["Urination_high"] = df_urination_high["Urination"].apply(lambda x: 1 if x >2.45 else 0)
df_urination_high = df_urination_high.drop("Urination", axis=1)

df_urination_high["Urination_high"].value_counts()

cat_variables = ["Partial Paresis_yes", "Urination_high", "Itching_yes", "Alopecia_yes", "Muscle Stiffness_yes",
                   "Polyphagia_yes", "Weakness_yes", "Visual Blurring_yes", "Sudden Weight Loss_yes", "Polydipsia_yes",
                   "Gender_male", "Obesity_yes", "TCep_yes", "Delayed Healing_yes", "Genital Thrush_yes",
                    "Irritability_yes"]
numeric_variables = ["BMI", "Temperature", "Weight", "Height", "Age"]

df = df_urination_high.copy()

X = df.drop("Diabetes_positive", axis=1)
y = df_urination_high["Diabetes_positive"]
X_cat = X[cat_variables]
X_numeric = X[numeric_variables]

def corr_test(X,y,score_type,nr_features="all"):
  # define feature selection
  fs = SelectKBest(score_func=score_type, k=nr_features)
  # apply feature selection
  X_selected = fs.fit_transform(X, y)

  dict_values = dict()
  dict_values[score_type.__name__ +"-score"] = np.round(fs.scores_,1)
  dict_values["p-value"] = np.round(fs.pvalues_,4)

  return pd.DataFrame(data = dict_values, index = fs.feature_names_in_).sort_values(by = "p-value")

print(f'Categorical features vs categorical target \n\n {corr_test(X_cat,y,chi2)}')

print(f'Numeric features vs categorical target \n\n {corr_test(X_numeric,y,f_classif)}')

sns.heatmap(X_numeric.corr(), annot = True, cmap='YlGnBu')      # Pearson's correlation coefficient - Numeric vs Numeric
plt.show()

for name in X_numeric:
    print("\n")
    print(name)
    frame = corr_test(X_cat, X_numeric[name], f_classif)
    print(frame[frame["p-value"] < 0.05])
    
for name in X_cat:
    print("\n")
    print(name)
    frame = corr_test(X_cat, X_cat[name], chi2)
    print(frame[frame["p-value"] < 0.05])
    
df.plot.scatter("Weight", 'BMI', c = ['r' if t == 1 else 'b' for t in diabetes_df["Diabetes_positive"]])
plt.show()

sns.pairplot(data=df[numeric_variables])
plt.show()

corr_numcat = pd.DataFrame(columns = numeric_variables, index = cat_variables)

for name in cat_variables:
    frame = corr_test(X_numeric, X_cat[name], f_classif)
    corr_numcat.loc[name] = frame["p-value"]

corr_numcat = corr_numcat[corr_numcat.columns].astype(float)
fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(corr_numcat, annot = True, ax = ax)

corr_cat = pd.DataFrame(columns = cat_variables, index = cat_variables)
for name in X_cat:
    frame = corr_test(X_cat, X_cat[name], chi2)
    corr_cat.loc[name] = frame["p-value"]

corr_cat = corr_cat[corr_cat.columns].astype(float)
fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(corr_cat, annot = True, ax = ax)
plt.show()

# TASK 3 

crosstab = pd.crosstab(diabetes_df['TCep_yes'], diabetes_df['Diabetes_positive']) 
print(crosstab)

c, p, dof, expected = chi2_contingency(crosstab)
print("P-value:", p)

print("Ratio of patients with Diabetes in each Occupation")
df[['Occupation','Diabetes_positive']].groupby(['Occupation']).sum()/df[['Occupation','Diabetes_positive']].groupby(['Occupation']).count()

print("Number of total patients in each Occupation")
df[['Occupation','Diabetes_positive']].groupby(['Occupation']).count()


# TASK 4
from sklearn.model_selection import train_test_split
from sklearn import metrics

features = ['Alopecia_yes', 'Gender_male', 'Irritability_yes', 'Partial Paresis_yes', 'Polyphagia_yes', 'Sudden Weight Loss_yes', 'Visual Blurring_yes', 'Weakness_yes', 'Weight', 'Urination_high']
labels_df = df[['Diabetes_positive']]
features_df = df[features]

x_train, x_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.3, random_state=1)

from itertools import combinations

# make a list of all combinations of
def generate_combinations(feature_list, n):
    tuple_list = list(combinations(feature_list, n))
    res = [list(el) for el in tuple_list]
    return res

def generate_combinations_one_to_n(feature_list):
    n = len(feature_list)
    res = []
    for i in range(1, n+1):
        res += generate_combinations(feature_list, i)
    return res

print(features)
combs = generate_combinations_one_to_n(features)

print("Possible feature combinations using 1 to 10 features: ", len(combs))
print(combs[25])
# We can now get the values of our feature dataframe with a specific combination by simply indexing, like this:
#print(features_df[combs[25]])

# Example for a single combination from this list
from sklearn import svm

curr = 25
X_train, X_test, y_train, y_test = train_test_split(features_df[combs[curr]], labels_df, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Create and fit classifier
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train, np.ravel(y_train))

y_pred = svm_clf.predict(X_test)

def calculate_metrics(y_test, y_pred):
    res = {'accuracy': metrics.accuracy_score(y_test, y_pred),
           'precision': metrics.precision_score(y_test, y_pred),
           'recall': metrics.recall_score(y_test, y_pred),
           'f1': metrics.fbeta_score(y_test, y_pred, beta=1),
           'f2': metrics.fbeta_score(y_test, y_pred, beta=2)}

    return res

def print_metrics(metric_dict, main_only=False):
    if not main_only:
        print("Accuracy:  %.3f" % metric_dict['accuracy'])
        print("Precision: %.3f" % metric_dict['precision'])
        print("Recall:    %.3f" % metric_dict['recall'])
        print("F1-score:  %.3f" % metric_dict['f1'])
    print("F2-score:  %.3f" % metric_dict['f2'])

my_metrics = calculate_metrics(y_test, y_pred)
print_metrics(my_metrics, main_only=True)

def calculate_fairness(gender_df, X, y, clf):
    
    y_hat = clf.predict(X)
    
    metric_df = gender_df[['Gender_male']]
    metric_df['labels'] = y[['Diabetes_positive']]
    metric_df['prediction'] = y_hat
    
    male_fn = len(metric_df[(metric_df['Gender_male']==1) & (metric_df['labels']==1) & (metric_df['prediction']==0)])
    female_fn = len(metric_df[(metric_df['Gender_male']==0) & (metric_df['labels']==1) & (metric_df['prediction']==0)])

    male_fp = len(metric_df[(metric_df['Gender_male']==1) & (metric_df['labels']==0) & (metric_df['prediction']==1)])
    female_fp = len(metric_df[(metric_df['Gender_male']==0) & (metric_df['labels']==0) & (metric_df['prediction']==1)])

    male_tp = len(metric_df[(metric_df['Gender_male']==1) & (metric_df['labels']==1) & (metric_df['prediction']==1)])
    female_tp = len(metric_df[(metric_df['Gender_male']==0) & (metric_df['labels']==1) & (metric_df['prediction']==1)])

    male_tn = len(metric_df[(metric_df['Gender_male']==1) & (metric_df['labels']==0) & (metric_df['prediction']==0)])
    female_tn = len(metric_df[(metric_df['Gender_male']==0) & (metric_df['labels']==0) & (metric_df['prediction']==0)])

    print("Male FN:", male_fn, "Female FN:", female_fn)
    print("Male FP:", male_fp, "Female FP:", female_fp)

    print("Male TN:", male_tn, "Female TN:", female_tn)
    print("Male TP:", male_tp, "Female TP:", female_tp)
    
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree 

# Linear Support Vector Machine
def create_SVM(**kwargs):
    svm_clf = svm.SVC(kernel='linear')
    return svm_clf

# A small feed-forward (Multi-Layer-Perceptron) neural network
def create_simple_MLP(**kwargs):
    mlp_clf = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=42, max_iter=200, **kwargs)
    return mlp_clf

# A larger feed-forward (Multi-Layer-Perceptron) neural network
def create_large_MLP(**kwargs):
    mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 500, 500, 20), random_state=42, max_iter=500, learning_rate='adaptive', **kwargs)
    return mlp_clf

# k-Nearest Neighbors, k=30
def create_KNN(**kwargs):
    knn_clf = KNeighborsClassifier(n_neighbors=30, **kwargs)
    return knn_clf

# Decision Tree
def create_tree(**kwargs):
    tree_clf = tree.DecisionTreeClassifier(**kwargs)
    return tree_clf

from tqdm import tqdm

@ignore_warnings(category=ConvergenceWarning)
def predict_evaluate_all_combinations(feat_list, feat_df, lab_df, clf_creator):
    # All possible combinations using 1 to n features
    combs = generate_combinations_one_to_n(feat_list)
    
    X_train_, X_test_, y_train_, y_test_ = train_test_split(feat_df, lab_df, test_size=0.2, random_state=42)

    # Initial values
    best_performance = 0.0
    best_index = 0
    best_comb = combs[0]
    best_metrics = {}
    best_clf = ""
    best_data = []

    # Iterate every combination
    for i, comb in enumerate(tqdm(combs)):
        #print("i=%3d, feature set: %s" % (i, comb))
        X_train, X_test, y_train, y_test = train_test_split(feat_df[comb], lab_df, test_size=0.2, random_state=42)

        # Create and fit SVM
        clf = clf_creator()
        clf.fit(X_train, np.ravel(y_train))

        # Make predictions
        y_pred = clf.predict(X_test)

        # Calculate and get metrics
        res_metrics = calculate_metrics(y_test, y_pred)
        #print_metrics(res_metrics, main_only=True)

        if res_metrics['f2'] > best_performance:
            best_performance = res_metrics['f2']
            best_index = i
            best_comb = comb
            best_metrics = res_metrics
            best_clf = clf
            best_data = [X_train, X_test, y_train, y_test]
    
        #if any(value == 1 for value in res_metrics.values()):
        #    print("-- WARNING: some metric is exactly 1! Features: ", comb, "metrics: ", res_metrics)
        #if res_metrics['recall'] == 1:
        #    print("hey", comb)

    print("Top performance: %.3f, at index %3d, feature set: %s" % (best_performance, best_index, best_comb))
    print_metrics(best_metrics)
    calculate_fairness(X_test_, best_data[1], best_data[3], best_clf)
    return best_clf, best_comb, best_performance, best_metrics, best_data

print("Using a SVM classifier:")
svm_clf, svm_comb, svm_perf, svm_metrics, svm_data = predict_evaluate_all_combinations(features, features_df, labels_df, create_SVM)

# print("Using a simple feedforward neural net (MLP) classifier:")
# ffsimple_clf, ffsimple_comb, ffsimple_perf, ffsimple_metrics, ffsimple_data = predict_evaluate_all_combinations(features, features_df, labels_df, create_simple_MLP)

# print("Using a larger feedforward neural net (MLP) classifier:")
# fflarge_clf, fflarge_comb, fflarge_perf, fflarge_metrics, fflarge_data = predict_evaluate_all_combinations(features, features_df, labels_df, create_large_MLP)

print("Using a KNN with K=30:")
knn_clf, knn_comb, knn_perf, knn_metrics, knn_data = predict_evaluate_all_combinations(features, features_df, labels_df, create_KNN)

print("Using a tree classifier:")
tree_clf, tree_comb, tree_perf, tree_metrics, tree_data = predict_evaluate_all_combinations(features, features_df, labels_df, create_tree)

fix, ax = plt.subplots(figsize=(40,20))
tree.plot_tree(tree_clf)
plt.show()

# Anonymize a single row using differential privacy: default is flipping a fair coin,
# which gives ln(3)-differential privacy
def anonymize_row(row, theta=0.5):
    # Decide whether to change or not
    coin = np.random.choice([True, False], p=(theta, 1-theta))
    changed = False

    if coin != True: # generate responses randomly
        for idx, val in row.items():
            # generate single randomized entry
            new_val = np.random.choice([1, 0], p=(theta, 1-theta))
            row[idx] = new_val
        changed = True

    return row, changed

# Anonymize entire dataframe using differential privacy
def anonymize_df(df, verbose=False):
    res_df = df.copy()
    num_changed = 0

    for i, row in res_df.iterrows():
        # some issue causes the loop to go too far, so we break here
        if i == len(res_df):
            if verbose:
                print("num changed:", num_changed)
            break
        # anonymize current row
        new_row, changed = anonymize_row(row, theta=0.5)
        if changed:
            num_changed += 1

        # assign the generated row to the output DataFrame
        res_df.iloc[i] = new_row

    return res_df

# Given a classifier, feature selection, training and test data and some original metrics,
# make a differentially private data set and do prediction based on this,
# finally reporting the original versus current metrics
def anonymize_predict_report(feat_comb, df, data, clf_creator, orig_metrics, verbose=False):
    #X_train, X_test, y_train, y_test = data

    df_anonymized = anonymize_df(df, verbose=verbose)

    X_train, X_test, y_train, y_test = train_test_split(df_anonymized, labels_df, test_size=0.2, random_state=42)

    new_clf = clf_creator()
    new_clf.fit(X_train, np.ravel(y_train))

    y_pred= new_clf.predict(X_test)
    print(y_pred)

    new_metrics = calculate_metrics(y_test, y_pred)

    clf_type = clf_creator.__name__.split("_")[1]

    print("Original metrics for %s classifier:" % clf_type)
    print_metrics(orig_metrics)
    print()
    print("New metrics for %s classifier:" % clf_type)
    print_metrics(new_metrics)
    print("Using features: %s", feat_comb)


svm_df = features_df[svm_comb]
anonymize_predict_report(svm_comb, svm_df, svm_data, create_SVM, svm_metrics, verbose=True)

# ffsimple_df = features_df[ffsimple_comb]
# anonymize_predict_report(ffsimple_comb, ffsimple_df, ffsimple_data, create_simple_MLP, ffsimple_metrics)

# fflarge_df = features_df[fflarge_comb]
# anonymize_predict_report(fflarge_comb, fflarge_df, fflarge_data, create_large_MLP, fflarge_metrics)

knn_df = features_df[knn_comb]
anonymize_predict_report(knn_comb, knn_df, knn_data, create_KNN, knn_metrics)

tree_df = features_df[tree_comb]
anonymize_predict_report(tree_comb, tree_df, tree_data, create_tree, tree_metrics)
