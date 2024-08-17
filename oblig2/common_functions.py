import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, LeaveOneOut, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import shap

from classes import BMIImputer, FlexibleImputer, BinarizerWithNan, Debug, SelectFeatures


def verbose_print(description, count_before, count_after, verbose=False):
    """
    Print extra information only when the verbose flag is set.

    Used for data cleanup where some rows/columns are removed from a DataFrame, and
    will print the number of rows/columns based on before and after numbers.

    Parameters
    ----------
    description: str
        Description text to include in the line to be printed.
    count_before: int
        Number of individuals before a certain operation was performed. NB the function does not compute
        these numbers itself: the calling program is responsible for counting and providing this.
    count_after: int
        Number of individuals after a certain operation was performed. NB the function does not compute
        these numbers itself: the calling program is responsible for counting and providing this.
    verbose: bool
        This flag dictates whether anything is printed. When set to True this function will print
        the extra information. When set to False (default), nothing is printed by this function.

    Returns
    -------

    """
    if verbose:
        print("Cleanup: %s" % description)
        print("Number of individuals before: %d" % count_before)
        print("Number of individuals after:  %d" % count_after)


# 1. Data cleanup
def read_cleanup_dataset(filepath, verbose=False):
    """
    Perform various data cleanup setup, as a function.

    Performs the following cleanup:
    0. Read filename into DataFrame
    1. Lowercase all string values
    2. Remove individuals with nan in Age, Gender, Height, Weight, GP, Occupation
    3. Scale height from meters to cm, where needed
    4. Introduce BMI feature
    5. Drop rows according to the following criteria:
      a) Remove individuals with 2 or more nan-values
      b) Remove individuals with Age > 110, Age < 0
      c) Remove individuals with Urination > 4
      d) Remove individuals with BMI < 15
      e) Remove individuals with race != 'white'
    6. Drop Race column

    Parameters
    ----------
    filepath: str
        Path to CSV file containing whole dataset we wish to load into memory. Will be
        read into a DataFrame. This file can be an entirely untouched, raw dataset.
    verbose: bool
        Flag to print extra information to the terminal. Note that the script should print
        quite easy-to-understand details, evne when this flag isn't set. Default: False.

    """

    # Read file into DataFrame
    input_df = pd.read_csv(filepath)
    if verbose:
        print("Read file %s, shape of Dataframe is %s" % (filepath, input_df.shape))

    # 1. Lowercase all string values
    categorical_cols = input_df.describe(include='object').columns
    for i in categorical_cols:
        input_df[i] = input_df[i].str.lower()

    if verbose:
        print("Starting data cleanup...")
        print("Number of individuals before cleanup: %d" % len(input_df))

    # 2. Remove individuals with nan in Age, Gender, Height, Weight, GP, Occupation
    before = len(input_df)
    input_df.dropna(subset=["Age", "Gender", "Height", "Weight", "GP", "Occupation"], inplace=True)
    after = len(input_df)
    verbose_print("Drop rows with na in [Age, Gender, Height, Weight, GP, Occupation]:", before, after, verbose=verbose)

    # 3. Scale height from meters to cm, where needed
    # 2.75 as a threshold means even the world's tallest man would be scaled correctly
    input_df.loc[input_df['Height'] < 2.75, "Height"] = input_df['Height'] * 100

    # 4. Introduce BMI feature
    before = len(input_df)
    input_df["BMI"] = input_df["Weight"] / ((input_df["Height"] / 100) ** 2)
    after = len(input_df)
    verbose_print("Weight/height: introduce BMI feature", before, after, verbose=verbose)

    # 5. Drop rows according to criteria:
    #   a) Remove individuals with 2 or more nan-values
    before = len(input_df)
    na_rows = list(input_df.loc[input_df.isnull().sum(1) > 1].index)
    input_df = input_df.drop(na_rows)
    after = len(input_df)
    verbose_print("Drop individuals with >=2 nan-values", before, after, verbose=verbose)

    #   b) Remove individuals with Age above 110 or below 0
    before = len(input_df)
    input_df = input_df[(input_df['Age'] <= 110) & (input_df['Age'] >= 0) | input_df['Age'].isnull()]
    after = len(input_df)
    verbose_print("Age", before, after, verbose=verbose)

    #   c) Remove individuals with Urination > 4
    before = len(input_df)
    input_df = input_df[(input_df['Urination'] < 4) | input_df['Urination'].isnull()]
    after = len(input_df)
    verbose_print("Urination: remove >4", before, after, verbose=verbose)

    #   d) Remove individuals with BMI < 15
    before = len(input_df)
    input_df = input_df[(input_df['BMI'] >= 15) | input_df['BMI'].isnull()]
    after = len(input_df)
    verbose_print("BMI: remove <15", before, after, verbose=verbose)

    #   e) Remove individuals with race != 'white'
    before = len(input_df)
    input_df = input_df[(input_df["Race"] == "white")]
    after = len(input_df)
    verbose_print("Race: remove != white", before, after, verbose=verbose)

    # 6. Drop the Race column, due to too few individuals with non-white ethnicities
    before = len(input_df.columns)
    input_df = input_df.drop(['Race'], axis=1)
    after = len(input_df.columns)
    verbose_print("Drop column Race: columns", before, after, verbose=verbose)

    if verbose:
        print("Data cleanup complete!")
        print("Number of individuals after cleanup: %d" % len(input_df))

    return input_df


def make_preproc_pipeline(X, y, verbose=False, random_state=42):
    """ Preprocessing pipeline with the following steps:
        1. Binarize Urination s.t. Urination > 2.45 => Urination = 1, else Urination = 0
        2. Impute missing values in Obesity column based on BMI
        3. Binary columns: perform flexible imputation based on yes/no frequencies for each column
        4. Binary columns: do one-hot encoding

        Parameters
        ----------
        X : Pandas DataFrame of shape (n_samples, n_features).
            Features to process.
        y : {Pandas DataFrame or numpy-like column}.
            Target variable to process.
        verbose : bool, default=False.
                  If True: prints out details about each step performed.
        random_state: int, default=42.

        Returns
        -------
        full_pipeline : sklearn.pipeline.Pipeline object.
                        Includes all the proprecessing steps for our experiment.
        sorted_features : List[str].
                          List of all variables names in the new sorted order after
                          all pipeline steps have been performed.
    """

    if verbose:
        print("== Creating pre-processing pipeline...")

    # Steps 3-4: Flexible imputation of binary columns + one-hot encoding
    binary_pipeline = Pipeline(steps=[
        ('flexible impute', FlexibleImputer(random_state=random_state)),
        ('one-hot encoding', OneHotEncoder(drop='first', sparse=False)),
    ])

    # List of all non-binary variables
    not_binary_cols = ['Age', 'Occupation', 'GP', 'Height', 'Weight', 'Temperature', 'BMI']
    # List of all binary columns
    binary_cols = [col for col in X.columns if col not in not_binary_cols]
    # List of all variables in sorted order
    sorted_features = binary_cols + not_binary_cols

    # only for binary columns
    binary_impute_onehot = ColumnTransformer(transformers=[
        ('binaries', binary_pipeline, binary_cols)
    ], remainder='passthrough')

    # Steps 1-4
    preprocessing_pipeline = Pipeline(steps=[
        ('binarize urination', BinarizerWithNan("Urination", threshold=2.45)),
        ('impute obesity', BMIImputer("BMI", "Obesity")),
        ('impute binaries and onehot', binary_impute_onehot)
    ])

    if verbose:
        print("Steps of main pipeline:")
        for step in preprocessing_pipeline.steps:
            print("- ", step)

    return preprocessing_pipeline, sorted_features


def make_transformer_pipeline(preproc_pipeline, feat_select_pipeline, verbose=False):
    """ Creating the full transformer part of the pipeline by combining
        the preprocessing pipeline and feature selection pipeline.

        Parameters
        ----------
        preproc_pipeline : sklearn.pipeline.Pipeline object.
                           Includes all steps of the preprocessing pipeline
        feat_select_pipeline : sklearn.pipeline.Pipeline object.
                               Includes the step of the feature selection pipeline
        verbose : bool, default=False.
                  If True: prints out details about each step performed.

        Returns
        -------
        transformer_pipeline : sklearn.pipeline.Pipeline object.
                               Includes all the steps for the transformer part
                               in our experiment.
    """
    # Combining the two pipeline objects to one pipeline.
    transformer_pipeline = Pipeline(steps=[
        ('preprocessing', preproc_pipeline),
        ('feature selection', feat_select_pipeline)
    ])

    return transformer_pipeline


def onehot_encode_y(y, verbose=False):
    """A simple function to one-hot encode a single-column DataFrame,
    designed to be used for a target value.

    Parameters
    ----------
    y: pd.Series of length n_samples
        Data to be one-hot encoded.
    verbose: bool
        Set flag to print extra information from execution.

    Returns
    -------
    y: pd.Dataframe of shape (n_samples, 1)
        One-hot encoded data.

    """
    encoder = OneHotEncoder(drop='first', sparse=False)
    # Make pd.Series into pd.DataFrame
    y = pd.DataFrame(y, columns=["Diabetes"])

    if verbose:
        print("= y before:")
        print(y.head(5))

    # Perform the one-hot encoding
    y = encoder.fit_transform(y)
    # Produces an np.ndarray, so we re-cast it to a DataFrame
    y = pd.DataFrame(y, columns=["Diabetes"])

    if verbose:
        print("= y after:")
        print(y.head(5))

    return y


def make_feat_select_pipeline(X, sorted_features, features=None, verbose=False):
    """ Creating a feature selection pipeline, which is added to the transformer pipeline.
        Also yields feature names in sorted order to keep track of them.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features).
            Numpy array of our data after the preprocessing part.
        sorted_features : List[str].
                          List of all variable names sorted in the correct order.
        features : List[str], deafult=None.
                   The desired features to select from X. If None, a list of default
                   features are chosen.
        verbose : bool, default=False.
                  If True: prints out details about each step performed.

        Returns
        -------
        feature_pipeline : sklearn.pipeline.Pipeline object.
                           The steps performed in the feature selection.
        feature_names : List[str].
                        List of the variable names in sorted order after that we end up with.
    """

    if verbose:
        print("== Creating pre-processing pipeline...")
    # Imposing a list of default features to be chosen if none are provided
    if features is None:
        features = ['Alopecia', 'Gender', 'Irritability', 'Partial Paresis', 'Polyphagia', 'Sudden Weight Loss',
                    'Visual Blurring', 'Weakness', 'Weight', 'Urination']

    # The index of the chosen features in the numpy array
    index_of_relevant_features = [idx for idx, e in enumerate(sorted_features) if e in features]

    # The name of the chosen features in the numpy array
    feature_names = [sorted_features[i] for i in range(len(sorted_features)) if sorted_features[i] in features]

    # Defining the feature selection pipeline
    feature_pipeline = Pipeline(steps=[
        ("feature_selection", SelectFeatures(index_of_relevant_features))
    ])

    return feature_pipeline, feature_names


def scale_height(val):
    """Scale a (height) value given in meters (m) into centimeters (cm).

    Checks for the value, and scales anything that is plausibly a human height given in meters, into centimeters.
    Any values appearing to be measured in centimeters are returned as-is.

    Parameters
    ----------
    val: float
        value to scale from meters (m) to centimeters (cm)

    Returns
    -------
    val:
        scaled value, in centimeters (cm)

    """
    # If height is below 2.75, assume it is in meters and scale
    if val < 2.75:
        return val * 100
    return val


# 2. Baseline
def calculate_metrics(y_true, y_pred):
    """Given arrays of true and predicted values for y, calculate a selection
    of metrics and return as a dictionary.
    The metrics that are returned are:
    - 'accuracy'
    - 'precision'
    - 'recall'
    - 'f1' -> the F1 score
    - 'f2' -> the F2 score (f-score with beta=2, weighting recall as 2x more important than precision)

    Parameters
    ----------
    y_true: np.ndarray of shape (n_samples, 1)
        True (gold) y values, for n_samples number of individuals.
    y_pred: np.ndarray of shape (n_samples, 1)
        Predicted y values, for the same n_samples individuals.

    Returns
    -------
    res: dict
        Dictionary with relevant metrics calculated.
    """
    res = {'accuracy': metrics.accuracy_score(y_true, y_pred),
           'precision': metrics.precision_score(y_true, y_pred),
           'recall': metrics.recall_score(y_true, y_pred),
           'f1': metrics.fbeta_score(y_true, y_pred, beta=1),
           'f2': metrics.fbeta_score(y_true, y_pred, beta=2)}

    return res


def print_metrics(metric_dict, main_only=False):
    """Print metrics as returned by calculate_metrics().

    Parameters
    ----------
    metric_dict: dict
        Dict of metrics, as returned from calculate_metrics(). Has 'accuracy', 'precision', 'recall', 'f1', 'f2'.
    main_only: bool
        Print only the main metric, defined as the F2 score ('f2'). Default False.

    """
    if not main_only:
        print("Accuracy:  %.3f" % metric_dict['accuracy'])
        print("Precision: %.3f" % metric_dict['precision'])
        print("Recall:    %.3f" % metric_dict['recall'])
        print("F1-score:  %.3f" % metric_dict['f1'])
    print("F2-score:  %.3f" % metric_dict['f2'])


def create_evaluate_baseline_model(transformer_pipeline, model_class, X_train, y_train, X_test, y_test, verbose=False):
    """ Fits the provided model by using the full transform-fit method of sklearn pipeline and prints out
        the performance metrics of the fitted model.

        NOTE: the name indicates this is only for the baseline model (SVM), but it can be (and is)
        re-used to train different models. In our project, we also train a Logistic Regression model using this.

        Parameters
        ----------
        transformer_pipeline : sklearn pipeline
                               The transform part of the full transform-fit method of sklearn pipeline
        model_class : model object.
                      sklearn.svm.SVC or sklearn.linear_model.LogisticRegression
        X_train : Pandas DataFrame.
                  DataFrame with the data of the features used in training.
        y_train : Pandas series or numpy array like.
                  Column with data of the target variable used in training.
        X_test : Pandas DataFrame.
                 DataFrame with the data of the features used in testing.
        y_test : Pandas series or numpy array like.
                 Column with data of the target variable used in testing.
        verbose : bool, default=False.
                  If True: prints out details about each step performed.

        Returns
        -------
        full_pipeline : sklearn.pipeline.Pipeline object.
                        All steps performed using the transform-fit method from sklearn pipeline.
        y_pred_test : {Pandas series}
                      Column with all predicted values from the model provided (i.e. model_class)
        result_metrics : Dictionary.
                         Key-value pairs for each metric type. Keys: 'accuracy', 'precision', 'recall', 'f1', 'f2'.

    """
    if verbose:
        print("== Creating baseline model...")

    # transformer_pipeline does preprocessing and feature selection
    full_pipeline = Pipeline(steps=[
        ('transform', transformer_pipeline),
        ('model', model_class)
    ])

    if verbose:
        print("Baseline pipeline created!")
        for step in full_pipeline.steps:
            print("- ", step)

    # fits the model to the transformed (pipeline: preprocess and feature selected) training data
    full_pipeline.fit(X_train, y_train.values.ravel())
    # fits the model to the transformed (pipeline: preprocess and feature selected) test data
    y_pred_test = full_pipeline.predict(X_test)

    if verbose:
        print("Predictions are:")
        print(y_pred_test)

    # getting the performance metrics and printing them out
    result_metrics = calculate_metrics(y_test, y_pred_test)
    print("== Model created and evaluated! Metrics:")
    print_metrics(result_metrics)

    return full_pipeline, y_pred_test, result_metrics


# 4. Model evaluation
def f_score(recall, precision, beta):
    """A simple implementation of the F-score (sometimes known as "F-beta score").

    The beta parameter indicates a weighting of recall over precision: beta=1 (F1-score) gives equal weight,
    beta=2 considers recall 2x more important than precision, and beta=0.5 considers recall half as important.
    (This clearly extends to other values of beta as well, 2 and 0.5 are just given as examples.)

    Parameters
    ----------

    recall: List[float], length N
        List of recall values.
    precision: List[float], length N
        List of corresponding precision values.
    beta: float
        The beta parameter to the F-beta function, weighting

    Returns
    -------
    f_beta_scores: List[float], length N
        F-beta scores for the given individuals.
    """
    pairs = zip(recall, precision)

    f_beta_scores = []

    for recall, precision in pairs:
        f_beta = round((1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall), 3)

        f_beta_scores.append(f_beta)

    return f_beta_scores


def cv_eval(X_train, y_train, model, random_state=42, verbose=False):
    """Perform 5-fold cross-validation, given a model type.

    Parameters
    ----------
    X_train: pd.Dataframe of shape (n_samples, n_features)
        Feature set for our data.
    y_train: pd.Series of shape (n_samples, 1)
        Target variable for our data.
    model: sklearn classifier
        The model class to use for classification.
    random_state: int
        Seed for pseudo-random random generation.
    verbose: bool
        Set flag to print extra information during execution.
    """
    print("= Regular 5-fold Cross Validation:")

    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall'}

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_state)

    precision = []
    recall = []

    for i in scoring:
        score = scoring[i]
        accuracy = np.round(cross_val_score(model,
                                            X_train,
                                            y_train.values.ravel(),
                                            scoring=score,
                                            cv=cv),
                            3)

        if i == "precision":
            precision = accuracy
        elif i == "recall":
            recall = accuracy

        if verbose:
            print(f"cross validation {i} scores = ", accuracy)
        print(f'5 fold cross validation {i} = {round(np.mean(accuracy), 3)}. Range = {min(accuracy)} - {max(accuracy)}')

        if verbose:
            print("\n")

    f_beta1 = f_score(recall, precision, beta=1)
    f_beta2 = f_score(recall, precision, beta=2)

    if verbose:
        print(f"cross validation f1 scores = ", f_beta1)
    print(f"5 fold cross validation f1 score = {round(np.mean(f_beta1), 3)}. Range = {min(f_beta1)} - {max(f_beta1)}")

    if verbose:
        print("\n")
        print(f"cross validation f2 scores = ", f_beta2)

    print(f"5 fold cross validation f2 score = {round(np.mean(f_beta2), 3)}. Range = {min(f_beta2)} - {max(f_beta2)}")

    if verbose:
        print("\n")

    return


def vary_folds_cv_eval(X_train, y_train, model, random_state=42, verbose=False):
    """
    Run k-fold cross validation with the following number of folds: [3, 5, 10, 15, 20].

    Parameters
    ----------
    X_train: pd.DataFrame of shape (n_samples, n_features)
        Features for training data (to be split into folds).
    y_train: pd.Series of shape (n_samples, 1)
        Target variable for training data.
    model: sklearn classifier
        The model class to use for classification.
    random_state: int
        Seed for pseudo-random random generation.
    verbose: bool
        Set flag to print extra information during execution.
    """
    print("= Varying number of folds in cross validation:")

    scoring = {'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall'}

    folds = [3, 5, 10, 15, 20]

    f1 = []
    minf1 = []
    maxf1 = []
    f2 = []
    minf2 = []
    maxf2 = []
    recall = []
    precision = []

    for j in scoring:
        if j == "Accuracy":
            print(f"- {j}:")
        else:
            if verbose:
                print("\n")
                print(f"- {j}:")

        score = scoring[j]

        for i in folds:

            cv = RepeatedStratifiedKFold(n_splits=i, n_repeats=1, random_state=random_state)

            accuracy = np.round(cross_val_score(model,
                                                X_train,
                                                y_train.values.ravel(),
                                                scoring=score,
                                                cv=cv),
                                3)
            mean = np.mean(accuracy)

            if j == "Accuracy":
                print(f'Average {j} with {i} folds = %.3f,' % mean,
                      f"ranging from {min(accuracy)} - {max(accuracy)}")
            else:
                if verbose:
                    print(f'Average {j} with {i} folds = %.3f,' % mean,
                          f"ranging from {min(accuracy)} - {max(accuracy)}")

            if j == "Precision":
                precision = accuracy

            elif j == "Recall":
                recall = accuracy

                f1_score = f_score(recall, precision, beta=1)
                f1.append(np.mean(f1_score))
                minf1.append(min(f1_score))
                maxf1.append(max(f1_score))

                f2_score = f_score(recall, precision, beta=2)
                f2.append(np.mean(f2_score))
                minf2.append(min(f2_score))
                maxf2.append(max(f2_score))

    if verbose:
        print("\n")
        print("- F1-score:")
        for i in range(len(folds)):
            print(f"Average f1 score with {folds[i]} folds = %.3f," % np.mean(f1[i]),
                  f"ranging from {minf1[i]} - {maxf1[i]}")
        print("\n")

    print("- F2-score:")
    for i in range(len(folds)):
        print(f"Average f2 score with {folds[i]} folds = %.3f," % np.mean(f2[i]),
              f"ranging from {minf2[i]} - {maxf2[i]}")
    if verbose:
        print("\n")
    return


def loo_cv_eval(X_train, y_train, model, verbose=False):
    """Perform leave-one-out cross validation, for a given model.

    Parameters
    ----------
    X_train: pd.DataFrame of shape (n_samples, n_features)
        Features for training data (to be split into folds).
    y_train: pd.Series of shape (n_samples, 1)
        Target variable for training data.
    model: sklearn classifier
        The model class to use for classification.
    verbose: bool
        Set flag to print extra information during execution.
    """
    loo = LeaveOneOut()

    accuracy = cross_val_score(model,
                               X_train,
                               y_train.values.ravel(),
                               scoring="accuracy",
                               cv=loo)

    print("= Leave-One-Out cross validation")
    print('Leave one out mean accuracy = %.3f' % np.mean(accuracy))
    if verbose:
        print("\n")
    return


# 5. Interpretability
def describe_global_interpretability(full_pipeline, feature_selection, verbose=False):
    """Extracts feature coefficients and plots them to compare feature impact on the model.  

    Args:
        full_pipeline (sklearn pipeline): sklearn pipeline
        feature_selection (list): list containing the feature names in sorted order.
        verbose (bool, optional): If true, print out extra information regarding model coefficients. Defaults to False.
    """

    if verbose:
        print("Model coefficients:")
        print(feature_selection)
        print(full_pipeline.named_steps['model'].coef_)

    plt.barh(feature_selection, full_pipeline.named_steps['model'].coef_[0], align='center')
    plt.title("Feature Importance")
    plt.savefig('global_interpretability_logreg.png')
    plt.close('all')
    print("Feature Importance plot saved to current folder.")


def predict_and_convert(model, sample):
    """Converts binary numbers to class names.

    Args:
        model (sklearn model class): model to be used for making predictions.
        sample (pandas dataframe): data sample to be classified.

    Returns:
        string: returns the model prediction as a string. 
    """
    prediction = model.predict(sample.values.reshape(1, -1))
    if prediction[0] == 1:
        return "Diabetes Positive"
    elif prediction[0] == 0:
        return "Diabetes Negative"


def calculate_model_output(model_intercept, model_coefficients, sample):
    """Calculates P(X) for logistic regression.

    Args:
        model_intercept (int): model intercept value
        model_coefficients (list): list containing model coefficients.
        sample (pandas dataframe): sample from dataset.

    Returns:
        float: p(x), or the probability of having a positive diabetes diagnosis.
    """
    return 1 / (1 + np.exp(-(
                model_intercept + model_coefficients[0] * sample.iloc[0] + model_coefficients[1] * sample.iloc[1] +
                model_coefficients[2] * sample.iloc[2])))


def describe_example_individuals(X, full_pipeline, choose_random_sample):
    """Function for calculating model output for samples in the dataset.


    Args:
        X (pandas dataframe): dataframe containing the data samples
        full_pipeline (sklearn pipeline): sklearn pipeline
        choose_random_sample (bool): If True, new random samples will be selected from the dataframe to be used for local interpretability.
    """
    model = full_pipeline.named_steps['model']

    if choose_random_sample:
        samples = X.sample(n=3)
        sample1 = samples.iloc[0]
        sample2 = samples.iloc[1]
        sample3 = samples.iloc[2]
    else:
        sample1 = X.iloc[1]
        sample2 = X.iloc[10]
        sample3 = X.iloc[16]

    model_coefficients = full_pipeline.named_steps['model'].coef_[0]
    model_intercept = full_pipeline.named_steps['model'].intercept_[0]

    print("Sample 1:")
    print(sample1)
    print("Model output: ", calculate_model_output(model_intercept, model_coefficients, sample1))
    print("Prediction on Sample 1:", predict_and_convert(model, sample1))
    print("")
    print("Sample 2:")
    print(sample2)
    print("Model output: ", calculate_model_output(model_intercept, model_coefficients, sample2))
    print("Prediction on Sample 2:", predict_and_convert(model, sample2))
    print("")
    print("Sample 3:")
    print(sample3)
    print("Model output: ", calculate_model_output(model_intercept, model_coefficients, sample3))
    print("Prediction on Sample 3:", predict_and_convert(model, sample3))


# 5.2 SHAP
def shap_global_explainer(model, X, sorted_feature_names, model_type):
    """Calculates SHAP values for each feature and plots
    the average absolute SHAP value for each feature.

    Args:
        model (sklearn model class): sklearn model class to be explained
        X (dataframe): pandas dataframe containing data.
        sorted_feature_names (list): list containing the feature names in sorted order.
        model_type (str): string identifying the model. 
    """

    shap.initjs()
    explainer = shap.KernelExplainer(model.predict, data=X)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, plot_type="bar", features=sorted_feature_names, class_inds=[1], max_display=10,
                      show=False)
    plt.savefig(model_type + '_shap_summaryplot.png')
    plt.close('all')
    print("Summary plot saved to current folder.")


def shap_individual_explainer(model, X, sorted_feature_names, choose_random_sample):
    """Function to calculate shap values for individual samples and create
    force-plots to display how each feature contributes to a prediction 
    for individual samples. 

    Args:
        model (sklearn model class): sklearn model
        X (pandas dataframe): Dataframe containing the samples
        sorted_feature_names (list): list containing the feature names in sorted order.
        choose_random_sample (bool): If True, new random samples will be selected from the dataframe to be used for local interpretability.
    """
    explainer = shap.KernelExplainer(model.predict, data=X)
    shap_values = explainer.shap_values(X)

    if choose_random_sample:
        samples = X.sample(n=3)
        sample1 = samples.iloc[0]
        sample2 = samples.iloc[1]
        sample3 = samples.iloc[2]
    else:
        sample1 = X.iloc[1]
        sample2 = X.iloc[10]
        sample3 = X.iloc[16]

    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[1, :], features=sample1, matplotlib=True,
                    feature_names=sorted_feature_names, show=False)
    plt.savefig('shap_forceplot1.png')
    shap.force_plot(explainer.expected_value, shap_values[10, :], features=sample2, matplotlib=True,
                    feature_names=sorted_feature_names, show=False)
    plt.savefig('shap_forceplot2.png')
    shap.force_plot(explainer.expected_value, shap_values[16, :], features=sample3, matplotlib=True,
                    feature_names=sorted_feature_names, show=False)
    plt.savefig('shap_forceplot3.png')
    print("Force plots saved to current folder.")
