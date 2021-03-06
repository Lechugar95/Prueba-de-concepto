__title__ = ''
__author__ = 'Claudio Mori'
__credits__ = 'Sayo Makinwa'
__copyright__ = 'Copyright 2020, Thesis Project'

# coding=utf-8
import pandas as pd
import numpy as np
#import pdb
import os
import csv
# pdb.set_trace()

data = pd.read_csv('../raw_data/sha256_family.csv') #C:\ml\sapienza\ml\malware-analysis\raw_data

family_column = data["family"]
sha_column = data["sha256"]

FEATURES_SET = {
    "feature": 1,
    "permission": 2,
    "activity": 3,
    "service_receiver": 3,
    "provider": 3,
    "service": 3,
    "intent": 4,
    "api_call": 5,
    "real_permission": 6,
    "call": 7,
    "url": 8
}

def count_feature_set(lines):
    """
    Count how many features belong to a specific set
    :param lines: features in the text file
    :return:
    """
    features_map = {x: 0 for x in range(1, 9)}
    for l in lines:
        if l != "\n":
            set = l.split("::")[0]
            features_map[FEATURES_SET[set]] += 1
    features = []
    for i in range(1, 9):
        features.append(features_map[i])
    return features

feature_count = []
def read_sha_files():
    for filename in os.listdir('../raw_data/feature_vectors'):
        sha_data = open('../raw_data/feature_vectors/'+ filename)
        feature_count.append([filename] + count_feature_set(sha_data))
        sha_data.close()
    return feature_count

header = ['sha256', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8' ]
def create_csv_for_sha_data():
    with open("../processed_data/feature_vectors.csv", "wt", newline ='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(i for i in header)
        for j in read_sha_files():
            writer.writerow(j)

create_csv_for_sha_data()

feature_vectors_data = pd.read_csv('../processed_data/feature_vectors.csv')
sha256_data = feature_vectors_data['sha256']

"""
    map feature_vectors sha with it's corresponding
    output value from the sha_family file, when a file from
    feature_vectors is found, mark the file as a malware
    else, otherwise
"""
mask = np.in1d(sha256_data, sha_column)


# feature_vectors_data
output = pd.DataFrame({'output' : mask })
feature_vectors_data = feature_vectors_data.merge(output, left_index = True, right_index = True)
feature_vectors_data.to_csv('../processed_data/feature_vectors_data.csv')
