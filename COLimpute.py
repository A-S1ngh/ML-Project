# This file has a method which fills in missing values using COLimpute(filling in missing values using the column average)
import pandas as pd
import numpy as np

# Label each column for its respective feature
feature_names_1 = []
for i in range(1, 15):
    feature_names_1.append(f"Sample {i}")

feature_names_2 = []
for i in range(1, 51):
    feature_names_2.append(f"Sample {i}")

# Convert gene data to dataframe format
gene_data_1 = pd.read_csv("Data/MissingData1.txt", sep="\t", names=feature_names_1)
gene_data_2 = pd.read_csv("Data/MissingData2.txt", sep="\t", names=feature_names_2)


def COLimpute(dataset, output_file_name):
    # Work on seperate copy of dataset and replace missing values with NaN
    result_dataset = dataset.copy()
    result_dataset = result_dataset.replace(1e99, np.nan)

    # Replace NaN values with column means
    result_dataset = result_dataset.fillna(result_dataset.mean())
    print(result_dataset)
    return

    # Remove column names and output as txt file
    result_dataset.to_csv(
        f"Missing Value Answers/{output_file_name}.txt",
        header=None,
        index=None,
        sep="\t",
    )


COLimpute(gene_data_1, "gene_data_1_filled")
# COLimpute(gene_data_2, "gene_data_2_filled")
