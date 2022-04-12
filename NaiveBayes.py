import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import random
import time


def import_data(data_name, feature_count, labels):
    # Make feature/label names
    feature_names = []
    if labels:
        feature_names = ["Labels"]
    else:
        for i in range(1, feature_count + 1):
            feature_names.append(f"Feature {i}")

    # Read in data and return as dataframe
    if data_name == "TestData3":
        result_data = pd.read_csv(f"Data/{data_name}.txt", sep=",", names=feature_names)
    elif data_name == "TrainData2" or "TestData2":
        result_data = pd.read_csv(
            f"Data/{data_name}.txt", delim_whitespace=True, names=feature_names
        )
    else:
        result_data = pd.read_csv(
            f"Data/{data_name}.txt", sep="\t", names=feature_names
        )

    result_data = result_data.astype(float)
    result_data = result_data.abs()
    return result_data


def naive_bayes(
    train_data_name,
    train_label_name,
    test_data_name,
    feature_count,
    sample_count_test,
    target_count,
    output_file_name,
):
    # Import data
    train_data = import_data(train_data_name, feature_count, False)
    train_labels = import_data(train_label_name, feature_count, True)
    test_data = import_data(test_data_name, feature_count, False)

    # Fill in missing values
    train_data = train_data.replace(float("1.000000e+99"), np.nan)
    imputer = KNNImputer()
    train_data_filled = imputer.fit_transform(train_data)
    train_data = pd.DataFrame(train_data_filled, columns=list(train_data.columns))

    # Round values to make them categorical and find min and max for categorical summation
    train_data = train_data.round(decimals=1)
    train_data_min = min(train_data.min())
    train_data_max = max(train_data.max())

    # Feature Selection
    selector = SelectKBest(chi2, k=7)
    train_data_selected = selector.fit_transform(train_data, train_labels)
    cols = selector.get_support(indices=True)
    train_data = train_data.iloc[:, cols]

    # Join features and labels
    train_data = pd.concat([train_data, train_labels], axis=1)

    # Acquire Target Probabilities
    target_probabilities = train_data["Labels"].value_counts(normalize=True)
    tp_list = []
    if train_data_name == "TrainData5":
        for i in range(3, 9):
            tp_list.append(target_probabilities.at[i])
    else:
        for i in range(1, target_count + 1):
            tp_list.append(target_probabilities.at[i])

    # Create row and column indexes for probability dataframes
    prob_df_rows = []
    temp = train_data_min
    while temp < train_data_max + 0.1:
        prob_df_rows.append(round(temp, 1))
        temp += 0.1

    prob_df_columns = train_data.columns.delete(7)

    prob_df_list = []
    # Fill in dataframe probabilities
    for i in range(1, target_count + 1):
        prob_df = pd.DataFrame(columns=prob_df_columns, index=prob_df_rows)
        prob_df = prob_df.replace(np.nan, 0.0)
        target_df = train_data.loc[train_data["Labels"] == i]
        for j in prob_df_columns:
            row_prob = target_df[j].value_counts(normalize=True)
            for x in prob_df_rows:
                try:
                    prob = row_prob.at[x]
                except KeyError:
                    continue
                prob_df[j][x] = prob
        prob_df_list.append(prob_df)

    # Fill in missing values, feature select, and round out test data
    test_data = test_data.replace(float("1.000000e+99"), np.nan)
    imputer = KNNImputer()
    test_data_filled = imputer.fit_transform(test_data)
    test_data = pd.DataFrame(test_data_filled, columns=list(test_data.columns))
    test_data = test_data.iloc[:, cols]
    test_data = test_data.round(decimals=1)

    # calculate predicted labels
    target_prob_list = []
    for i in range(sample_count_test):
        target_prob_list.append([])

    for i in range(target_count):
        prob = tp_list[i]
        for row in range(sample_count_test):
            row_prob = []
            for column in test_data.columns:
                try:
                    test_value = test_data[column][row]
                    curr_prob = prob_df_list[i][column][test_value]
                    row_prob.append(curr_prob)
                except KeyError:
                    continue
            target_prob_list[row].append(np.prod(row_prob))

    # Format Labels
    prediction_labels = []
    for i in range(len(target_prob_list)):
        m = target_prob_list[i]
        maximus = np.amax(m)
        if maximus == 0.0:
            pl = random.randint(1, target_count + 1)
        else:
            pl = np.where(m == maximus)[0][0] + 1
        prediction_labels.append(pl)

    with open(f"{output_file_name}", "w") as filehandle:
        for listitem in prediction_labels:
            filehandle.write("%s\n" % listitem)


start = time.time()
naive_bayes(
    "TrainData1",
    "TrainLabel1",
    "TestData1",
    3312,
    53,
    5,
    "SinghSchuhClassification1.txt",
)
naive_bayes(
    "TrainData3",
    "TrainLabel3",
    "TestData3",
    13,
    2693,
    9,
    "SinghSchuhClassification3.txt",
)
naive_bayes(
    "TrainData2",
    "TrainLabel2",
    "TestData2",
    9182,
    74,
    11,
    "SinghSchuhClassification2.txt",
)
naive_bayes(
    "TrainData4",
    "TrainLabel4",
    "TestData4",
    112,
    1092,
    9,
    "SinghSchuhClassification4.txt",
)
naive_bayes(
    "TrainData5",
    "TrainLabel5",
    "TestData5",
    11,
    480,
    6,
    "SinghSchuhClassification5.txt",
)
end = time.time()
print(f"Time Taken = {end - start}")
