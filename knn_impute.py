from math import sqrt
import pandas as pd
import numpy as np
import scipy.spatial as sp


def train_to_df(file, col_num):
    df = pd.read_csv(file, header=None, names=range(col_num), delim_whitespace=True)
    return df


def label_to_ls(file):
    with open(file, "r") as f:
        lst = f.readlines()
        lst = [int(n) for n in lst]
    return lst


def knn(my_row, sample, x, k, na_ls):
    distance = []
    avgs = {}
    k_near = 0
    print(my_row)
    for s in sample.index:
        dist = np.nansum((my_row.values - dft.loc[[s]].values) ** 2)

        distance.append(dist)

    part = np.argpartition(distance, k)
    idx = sample.index[part[:k]]

    for n in na_ls:
        if n not in avgs.keys():
            sum = 0
            for i in idx:
                sum += dft[n][i]
            avg = sum / (dft.shape[0])

            avgs[n] = avg
    return avgs


def find_missing(my_df):
    new_df = my_df.copy()

    s = new_df.stack(dropna=False)
    d = {}
    for x in s.index[s.isna()]:
        a = list(x)

        if a[0] not in d.keys():
            d[a[0]] = []
        d[a[0]].append(a[1])

    return d


def prep_for_knn(d, test_file, k, out_file):
    results = dft.copy()
    size = len(d)
    for x in range(size):
        my_list = dft.columns.values.tolist()
        ls = d[x]
        new_df = dft.dropna(axis="rows", subset=ls)
        new_df = new_df.fillna(new_df.mean())
        new_df = new_df.drop(columns=ls, axis="columns")
        my_row = dft.loc[[x]]
        # my_row = my_row.drop(columns=ls, axis="columns")
        avgs = knn(my_row, new_df, x, k, ls)
        print(avgs)
        for key in avgs:
            print([key, x])
            results[key][x] = avgs.get(key)
            print(results[key][x])
    results.to_csv(
        f"gene-values-filled/{out_file}.txt",
        header=None,
        index=None,
        sep="\t",
    )
    print(results)
    pass


train_files = [f"data/MissingData{n}.txt" for n in range(1, 4)]
test_files = [f"data/TestData{n}.txt" for n in range(1, 4)]
train_label = [f"data/TrainLabel{n}.txt" for n in range(1, 4)]
out_files = [f"out_file{i}" for i in range(1, 7)]

feat_nums = [242, 758, 273]
sample_nums = [14, 50, 79]

for index in range(3):
    print(train_files[index])
    df = train_to_df(train_files[index], sample_nums[index])
    label = label_to_ls(train_label[index])
    df = df.replace(1e99, np.nan)
    dft = df.transpose()
    print(dft)

    fm = find_missing(dft)
    prep_for_knn(fm, label, 3, out_files[index])
