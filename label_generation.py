import random
from openai import OpenAI
import httpx
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import time
import re


def ini_client():
    client = OpenAI()
    return client

def chat(prompt, client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={ "type": "json_object" },
        messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": prompt}
        ]
    )
    response_origin = completion.choices[0].message.content
    # print(f"Original response: {response_origin}")
    return response_origin

def load_dataset(data_path, data, use_large):
    # data_file_list = os.listdir(data_path) # ['large.jsonl', 'small.jsonl']
    # print(data_file_list)
    data_file = os.path.join(data_path, data, "large.jsonl") if use_large else os.path.join(data_path, data, "small.jsonl")
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

def prompt_construct_generate_label(sentence_list, given_labels):
    json_example = {"labels": ["label name", "label name"]}
    prompt = f"Given the labels, under a text classicifation scenario, can all these text match the label given? If the sentence does not match any of the label, please generate a meaningful new label name.\n \
            Labels: {given_labels}\n \
            Sentences: {sentence_list} \n \
            You should NOT return meaningless label names such as 'new_label_1' or 'unknown_topic_1' and only return the new label names, please return in json format like: {json_example}"
    return prompt

def prompt_construct_merge_label(label_list):
    json_example = {"merged_labels": ["label name", "label name"]}
    prompt = f"Please analyze the provided list of labels to identify entries that are similar or duplicate, considering synonyms, variations in phrasing, and closely related terms that essentially refer to the same concept. Your task is to merge these similar entries into a single representative label for each unique concept identified. The goal is to simplify the list by reducing redundancies without organizing it into subcategories or altering its fundamental structure. \n"
    prompt += f"Here is the list of labels for analysis and simplification::{label_list}.\n"
    prompt += f"Produce the final, simplified list in a flat, JSON-formatted structure without any substructures or hierarchical categorization like: {json_example}"
    return prompt

def get_sentences(sentence_list):
    sentence = []
    for i in sentence_list:
        sentence.append(i['input'])
    return sentence

def label_generation(args, client, data_list, chunk_size):
    count = 0
    all_labels = []
    with open(args.given_label_path, 'r') as f: # load the given data
        given_labels = json.load(f)
    for label in given_labels[args.data]:
        all_labels.append(label)
    for i in range(0, len(data_list), chunk_size):
        sentence_list = data_list[i:i+chunk_size]
        sentences = get_sentences(sentence_list)
        prompt = prompt_construct_generate_label(sentences, given_labels[args.data])
        # print(f"prompt_length: {len(prompt)}")
        origin_response = chat(prompt, client)
        if origin_response is None:
            continue
        count += 1
        try:
            response = eval(origin_response)
        except:
            continue
        if isinstance(response[list(response.keys())[0]], list):
            current_labels = response[list(response.keys())[0]]
            for label in current_labels:
                if "unknown_topic" in label or "new_label" in label:
                    continue
                if label not in all_labels:
                    all_labels.append(label)
        else:
            all_labels.append(response[list(response.keys())[0]])
        
        if args.print_details:
            print(f"prompt: \n {prompt}")
            print(f"origin response: {origin_response}")
            print(f"response: {response}")
            print(f"length of labels: {len(response[list(response.keys())[0]])}")
            if count >= args.test_num:
                break
    return all_labels

def merge_labels(args, all_labels, client): 
    prompt = prompt_construct_merge_label(all_labels)
    response = chat(prompt, client)
    try:
        response = eval(response)
        merged_labels = []
        for key, sub_label_list in response.items():
            for label in sub_label_list:
                merged_labels.append(label)
        return merged_labels
    except:
        return all_labels


def write_dict_to_json(args, input, output_path, output_name):
    size = "large" if args.use_large else "small"
    file_name = os.path.join(output_path, '_'.join([args.data, size, output_name]) + ".json")
    with open(file_name, 'w') as json_file:
        json.dump(input, json_file, indent=2)
    print(f"JSON file '{file_name}' written.")

def main(args): 
    # 在分类时给定部分label（1/5），让模型根据已有的label生成新的label
    # 最后输出三个json，分别是原始的聚类label和模型总结完的分类label以及最后进行merge之后，用于后续分类或者计算相似度
    print("use_large: ", args.use_large)
    start_time = time.time()
    client = ini_client()
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    random.shuffle(data_list)
    label_list = get_label_list(data_list) # true labels
    print(f"Total cluster num: {len(label_list)}")
    write_dict_to_json(args, label_list, args.output_path, "true_labels")
    print(sorted(label_list))
    all_labels = label_generation(args, client, data_list, args.chunk_size)
    print(f"Total labels given by LLM: {len(all_labels)}")
    print(all_labels)
    write_dict_to_json(args, all_labels, args.output_path, "llm_generated_labels_before_merge")
    final_labels = merge_labels(args, all_labels, client)
    write_dict_to_json(args, final_labels, args.output_path, "llm_generated_labels_after_merge")
    print(f"Label number after merge: {len(final_labels)}")
    print(final_labels)
    print(f"Total time usage: {time.time() - start_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--given_label_path", type=str, default="./generated_labels/chosen_labels.json")
    parser.add_argument("--output_file_name", type=str, default="test.json")
    parser.add_argument("--use_large", action="store_true", help="Use large model if set, otherwise use small model") # True - Large; False - Small
    parser.add_argument("--print_details", type=bool, default=False) # print details
    parser.add_argument("--test_num", type=int, default=5) # how many test numbers
    parser.add_argument("--chunk_size", type=int, default=15) # how many sentences per chat
    parser.add_argument("--api_key", type=str, default="sk-", help="set the key to your OpenAI Key")
    args = parser.parse_args()
    main(args)