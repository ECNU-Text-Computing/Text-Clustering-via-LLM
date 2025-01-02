import json
import random
import os
import math

def find_sorted_folders(directory):
    folders = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            folders.append(entry.name)
    folders.sort()
    return folders

def load_dataset(data_path, data):
    data_file = os.path.join(data_path, data, "small.jsonl")
    print(f"Use dataset {data_file}")
    with open(data_file,'r') as f:
        data_list = []
        for line in f:
            json_object = json.loads(line)
            data_list.append(json_object)
    print(f"Length of dataset: {len(data_list)}")
    return data_list

def get_label_list(data_list):
    label_list = []
    for data in data_list:
        if data["label"] not in label_list:
            label_list.append(data["label"])
    return label_list


def main(): # seltect 20% of labels to be given to the LLMs
    data_path = "./dataset/"
    # print(find_sorted_folders(data_path))
    total_chosen_labels = dict()
    for data in find_sorted_folders(data_path):
        # total_chosen_labels[data] = []
        data_list = load_dataset(data_path, data)
        data_labels = get_label_list(data_list)
        print(len(data_labels))
        # choose_num = len(data_labels) // 5
        choose_num = int(0.2 * len(data_labels))
        print(f"Choose num: {choose_num}")
        total_chosen_labels[data] = random.choices(data_labels, k = choose_num)
    # print(total_chosen_labels)
    with open("./generated_labels/chosen_labels.json", 'w') as f:
        json.dump(total_chosen_labels, f, indent = 2)
    print(f"Write chosen labels to ./generated_labels/chosen_labels.json")


if __name__ == "__main__":
    main()