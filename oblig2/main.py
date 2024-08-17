from argparse import ArgumentParser

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from common_functions import *

if __name__ == '__main__':
    # Parser for command-line arguments
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", help="Path to raw dataset csv file. NB: if --train or --test are used, "
                                             "these take priority and this argument is ignored.",
                        default="data/diabetes.csv")
    parser.add_argument("--train", help="Path to training set csv file", default=None)
    parser.add_argument("--test", help="Path to test set csv file", default=None)
    parser.add_argument("-v", "--verbose", help="Print verbose output.", action='store_true', default=False)
    parser.add_argument("-r", "--random-state", help="set random seed (default = 42)", default=42)
    parser.add_argument("-s", "--random-samples", help="specify to randomly draw samples for local interpretability", action='store_true')
    args = parser.parse_args()

    # Load parsed args into variables
    infile_path = args.file
    train_path = args.train
    test_path = args.test
    verbose = args.verbose
    choose_random_sample = args.random_samples
    random_state = int(args.random_state)
    print("== Random seed is set to: %d" % random_state)

    # Instantiate model
    svm_model = SVC(kernel='linear')

    # If --train and --test aren't set, we use the raw dataset from --file
    # If --train and --test are set, these take priority
    if train_path is None and test_path is None:
        if infile_path == "data/diabetes.csv":
            print("== Using default dataset")
        print("== Input dataset file: %s" % infile_path)

        # Performs the following steps:
        # 0. Read filename into DataFrame
        # 1. Lowercase all string values
        # 2. Remove individuals with nan in Age, Gender, Height, Weight, GP, Occupation
        # 3. Scale height from meters to cm, where needed
        # 4. Introduce BMI feature
        # 5. Drop rows according to certain criteria (see method)
        # 6. Drop Race column
        # NB: as these modify the dimensions of the dataset, we see no way to implement these transforms
        # using sklearn Pipelines
        diabetes_df = read_cleanup_dataset(infile_path, verbose=verbose)
        print("== read_cleanup_dataset done!")

        # Split into features and target
        X = diabetes_df.drop("Diabetes", axis=1)
        y = diabetes_df["Diabetes"]

        # Train-test split (note seed!)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # Save train-test split to file
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        train.to_csv("data/train.csv", index=False)
        test.to_csv("data/test.csv", index=False)

    else:  # train and test paths are set
        # Read dataframes
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(train_path)

        # Split to X, y
        X_train = train_df.drop("Diabetes", axis=1)
        y_train = train_df["Diabetes"]

        X_test = test_df.drop("Diabetes", axis=1)
        y_test = test_df["Diabetes"]

    # Preprocessing pipeline:
    # 1. Binarize Urination s.t. Urination > 2.45 => Urination = 1, else Urination = 0
    # 2. Impute missing values in Obesity column based on BMI
    # 3. Binary columns: perform flexible imputation based on yes/no frequencies for each column
    # 4. Binary columns: do one-hot encoding
    preprocessing_pipeline, sorted_features = make_preproc_pipeline(X_train, y_train, verbose=verbose,
                                                                    random_state=random_state)
    print("== Pre-processing pipeline created!")
    # X_train = preprocessing_pipeline.fit_transform(X_train, y_train)

    # This is from project 1, the 3 features the best SVM model chose
    feature_selection = ['Gender', 'Irritability', 'Urination']
    feat_select_pipeline, sorted_feature_names = make_feat_select_pipeline(X_train, sorted_features,
                                                                           features=feature_selection)
    print("== Feature selection pipeline created!")

    transformer_pipeline = make_transformer_pipeline(preprocessing_pipeline, feat_select_pipeline, verbose=verbose)
    print("== Full transform pipeline created!")

    # One-hot encode target variable
    y_train = onehot_encode_y(y_train, verbose=verbose)
    y_test = onehot_encode_y(y_test, verbose=verbose)    

    # 2. Baseline
    # Baseline model: SVM
    # Note: the name "create_evaluate_baseline_model" indicates this is only for the baseline_model,
    # but it's also later re-used for training the Logistic Regression model.
    baseline_pipeline, preds, result_metrics = create_evaluate_baseline_model(
        transformer_pipeline, svm_model, X_train, y_train, X_test, y_test, verbose=verbose
    )

    # SVM model object to read f.ex. coefficients
    baseline_svm = baseline_pipeline.named_steps['model']

    # 3. Data leakage
    # NB: Covered in slides only!

    # 4. Model evaluation
    # Regular 5-fold cross validation
    print("== Further model evaluation")
    data_X = transformer_pipeline.fit_transform(X_train)
    X_train_clean = pd.DataFrame(data_X, columns=sorted_feature_names)
    X_test_clean = pd.DataFrame(transformer_pipeline.fit_transform(X_test), columns=sorted_feature_names)

    # Perform cross-validation and evaluate
    cv_eval(X_train_clean, y_train, SVC(kernel='linear'), random_state=random_state, verbose=verbose)

    # Varying number of folds in cross validation
    # We use 3, 5, 10, 15, and 20 folds
    vary_folds_cv_eval(X_train_clean, y_train, svm_model, random_state=random_state, verbose=verbose)

    # Leave-One-Out Cross Validation, requires clean X_train
    loo_cv_eval(X_train_clean, y_train, svm_model, verbose=verbose)

    # 5. Interpretability
    # Create and train a logistic regression model, and run inference
    logreg_model = LogisticRegression(penalty='none')
    # Note: the name "create_evaluate_baseline_model" indicates this is only for the baseline_model (SVM),
    # but it's re-used here for training the Logistic Regression model. Our baseline remains the SVM.
    baseline_pipeline, preds, result_metrics = create_evaluate_baseline_model(
        transformer_pipeline, logreg_model, X_train, y_train, X_test, y_test, verbose=verbose
    )       
    
    # Creates and saves a plot with the model coefficients
    describe_global_interpretability(baseline_pipeline, sorted_feature_names, verbose=verbose)

    # Compares model output for three randomly selected samples from the test set.
    # Set the choose_random_sample flag (from --random-samples argument) to draw these samples
    # actually randomly. Our default is to draw the same individuals every time
    describe_example_individuals(X_test_clean, baseline_pipeline, choose_random_sample)

    # 5.2 SHAP
    # Calculate SHAP values for global interpretability, Logistic Regression
    shap_global_explainer(
        baseline_pipeline.named_steps['model'], X_train_clean, sorted_feature_names, model_type='logreg'
    )
    
    # Calculate SHAP values for global interpretability, with SVM
    # This allows us to compare the two, see slides
    shap_global_explainer(baseline_svm, X_train_clean, sorted_feature_names, model_type='svm')
    
    # Calculating SHAP values for individual samples
    # Again, use --random-samples to draw fully randomly 3 individuals
    # Default is to pick the same three individuals every time
    shap_individual_explainer(
        baseline_pipeline.named_steps['model'], X_test_clean, sorted_feature_names, choose_random_sample
    )

    print("== All done!")
