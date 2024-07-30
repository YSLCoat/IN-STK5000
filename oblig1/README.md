# Assignment 1
## Group members: Espen H. Kristensen (espenhk), Cornelius Bencsik (corneb), Amir Basic (amirbas), Torstein Forseth (torfor)

### Video:
The video is available in this folder:
* High quality (~250MB): `video_high_quality.mp4`
* Low quality (~50MB): `video_low_quality.mp4`

### Slides:
Available from this folder as `Assignment1_slides.pdf` and `Assignment1_slides.pptx`.

### Code:
You can follow the code using the Jupyter notebook `main_notebook.ipynb` (recommended for testing), or
by running the Python script `diabetes_classifier.py` as follows:

~~~
python diabetes_classifier -f path_to_dataset.csv
~~~

For instance running it from this repo using the dataset from the `/data` folder:

~~~
python diabetes_classifier -f data/diabetes.csv
~~~

## Code breakdown:
The code more or less follows the order as presented in the slides/video, and mirrors more or less
perfectly the notebook. 

### Opening:
The opening section is simply a set of imports and reading the data, simple cleanup, as well as removing the rows with
two or more `NaN`-values. This portion ends at the comment `# PART 2`.

### Part 2 and 3: data analysis, grounds for feature selection
This starts at the comment `# PART 2`, and consists of the following. All of it follows the steps and order in the
slides and video:

1. Creating a detailed data overview. This can be printed by uncommenting the lines (around line 78)
~~~
result = pd.concat(frames)
printable = result.style.format()
print(result)
~~~
2. Creating a few boxplots for the columns for age, urination, weight and height
3. Removing the age outliers, as described in slides and video
4. Correcting the height measurements so they follow one scale, as described in slides and video
5. List and remove urination outliers, as described in slides and video
6. Consider creating a binary `Urination_high` with a cutoff at 2.45, and plotting and listing
list the ratios of diabetes patients in the patients that have high urination versus low urination (see point 13 below)
7. Creating a `BMI` variable, dropping low-weight outliers.
8. Listing non-white individuals in the `Race` column, and dropping these
9. Plotting `BMI` versus `Obesity`, and using this to fill in the missing value in the `Obesity` column.
10. Dropping entries with missing values in the columns `Age`, `Gender`, `Height`, `Weight`, `GP`, `Occupation`.
11. Creating the function `fill_na_df` to take a DataFrame and fill all missing values in every column with missing
values, by drawing randomly based on the frequencies of values for the filled in entries in that column. Then using
this to fill remaining columns with missing values.
12. Creating one-hot encodings of the binary categorical variables.
13. Creating a new binary column `Urination_high` using the cutoff described above, inserting this variable into our
DataFrame and removing the original `Urination` feature.
14. Looking at various forms of correlation tests, starting around the function `corr_test()`

### Part 4:
Starts around the comment `# PART 4`. Still following the structure of the slides and video, does the following:
1. Splits our main DataFrame into two: one `features_df` for just the features, and one `labels_df` for the target
variable only.
2. The functions `generate_combinations()` and `generate_combinations_one_to_n()` to allow creation of every possible
feature combination from 1 to n features, and generate a variable `combs` containing this.
3. Example code for training a linear SVM classifier for a specific one of these feature combinations.
4. Helper methods `calculate_metrics()` and `print_metrics()` to allow calculation and printing of relevant metrics,
including the flag `main_only` which lets you print the main feature (F2-score) only.
5. The helper method `calculate_fairness()` to calculate a fairness measure based on the rate of False Negatives and
False Positives for the `Gender_male` feature.
6. Helper methods `create_SVM()`, `create_simple_MLP()` etc to create various forms of classifiers, for later use.
7. The main method `predict_evaluate_all_combinations()` to take a list of features and a classifier creator method,
and train one classifier for each possible feature combination from 1 to n features. Finally select the one that
does the best on the main F2-score metric, and return out the relevant metrics and other components that will
be necessary later.
8. Training of 5 types of classifiers: SVM, simple MLP, large MLP, kNN and Decision Tree. **NB: if you uncomment
all these, particularly the large MLP will take some time to train (15-20 minutes on a moderately strong gaming PC)**.
9. The helper methods `anonymize_row()` and `anonymize_df()` to allow anonymization of a DataFrame through differential
privacy. **NOTE:** to get the same train-test split as the non-anonymized version above, we depend on setting the
same `random_state` for both. In production use this should be re-implemented such that you can do without the random
state, for final evaluation purposes.
10. The main `anonymize_predict_report()` method to anonymize a dataframe, create and train a classifier based on the
best feature set of that classifier (the non-anonymized version), and report the previous and new metrics. This will
also print the actual predictions, which demonstrates the issue we found with the linear SVM after anonymization
predicting all patients having diabetes. Note that this issue has been addressed in our video and slides.
