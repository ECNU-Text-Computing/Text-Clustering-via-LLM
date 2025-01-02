import argparse
import os
import json
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def load_data(data_path, data,  use_large):
    data_file = os.path.join(data_path, data, "large.jsonl") if use_large else os.path.join(data_path, data, "small.jsonl")
    with open(data_file,'r') as f:
        data_list = []
        for line in f:
            json_object = json.loads(line)
            data_list.append(json_object)
    return data_list

def load_predict_data(data_path, file_name):
    data_file = os.path.join(data_path, file_name)
    with open(data_file,'r') as f:
        data_dict = json.load(f)
    return data_dict

def get_labels(data_list):
    labels = []
    for data in data_list:
        labels.append(data["label"])
    return labels

def get_predict_labels(label_data_list, predict_data_dict):
    predict_labels = []
    for label_data in label_data_list:
        sentence = label_data["input"]
        for predict_label, sentence_list in predict_data_dict.items():
            if sentence in sentence_list:
                predict_labels.append(predict_label)
                break
    return predict_labels

def convert_label_to_ids(labels):
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    label_ids = [label_map[l] for l in labels]
    # print(label_ids)
    print(f"Length of labels: {len(labels)}")
    print(f"Number of Clusters: {n_clusters}")
    return np.asarray(label_ids), n_clusters


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    return {'ACC': clustering_accuracy_score(y_true, y_pred),
            'ARI': adjusted_rand_score(y_true, y_pred),
            'NMI': normalized_mutual_info_score(y_true, y_pred)}


def main(args):
    # print(args)
    label_data_list = load_data(args.data_path, args.data, args.use_large)
    labels = get_labels(label_data_list)
    print(f"total label length: {len(labels)}")
    
    predict_data_dict = load_predict_data(args.predict_file_path, args.predict_file)
    predict_labels = get_predict_labels(label_data_list, predict_data_dict)
    # print(len(predict_labels))
    
    labels = labels[:len(predict_labels)]

    print("Ground truth labels: ")
    y_true, cluster_true = convert_label_to_ids(labels = labels)
    print("Predict labels: ")
    y_pred, cluster_predict = convert_label_to_ids(labels = predict_labels)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    score = clustering_score(y_true=y_true, y_pred=y_pred)
    print(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--use_large", action="store_true", help="Use large model if set, otherwise use small model") # True - Large; False - Small
    parser.add_argument("--predict_file_path", type=str, default="./generated_labels/")
    parser.add_argument("--predict_file", type=str, default="") 
    args = parser.parse_args()
    main(args)