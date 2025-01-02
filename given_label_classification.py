import random
from openai import OpenAI
import httpx
import os
import json
import argparse
from tqdm import tqdm
import time

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

def get_predict_labels(output_path, data):
    data_file = os.path.join(output_path, data + "_small_llm_generated_labels_after_merge.json")
    
    with open(data_file, 'r') as f:
        data_list = json.load(f)
    data_list = list(set(data_list))
    return data_list

def prompt_construct(label_list, sentence):
    prompt = f"Given the label list and the sentence, please categorize the sentence into one of the labels.\n"
    prompt += f"Label list: {label_list}.\n"
    prompt += f"Sentence:{sentence}.\n"
    json_example = {"label_name": "label"}
    prompt += f"You should only return the label name, please return in json format like: {json_example}"
    return prompt

def answer_process(response,label_list):
    label = "Unsuccessful"
    try:
        response_new = eval(response)
    except:
        response_new = str(response)
    if isinstance(response_new, dict):
        total_label = []
        for key, value in response_new.items():
            total_label.append(value)
        for i in label_list:
            if i in total_label:
                label = i
                return label
    else:
        for i in label_list:
            if i in response_new:
                label = i
                return label
    return label

def known_label_categorize(args, client, data_list, label_list):
    answer = dict()
    length = args.test_num if args.print_details else len(data_list)
    answer["Unsuccessful"] = []
    for label in label_list:
        answer[label] = []
    for i in range(length):
        sentence = data_list[i]['input']
        prompt = prompt_construct(label_list, sentence)
        response = chat(prompt, client)
        if response is None:
            response_adjusted = "Unsuccessful"
        else:
            response_adjusted = answer_process(response, label_list)
        
        if response_adjusted in label_list:
            answer[response_adjusted].append(sentence)
        else:
            answer["Unsuccessful"].append(sentence)
        
        if args.print_details:
            print(f"---------------Sample {i + 1}-------------------")
            print(f"Question: {sentence}")
            print(f"prompt:\n{prompt}")
            print(f"Original Answer: {response}")
            print(f"Final Answer: {response_adjusted}")
            print(answer)
        if i % 200 == 0:
            print(f"Total sample number: {i}", end = "\t")
            write_answer_to_json(args, answer, args.output_path, args.output_file_name)
    return answer

def write_answer_to_json(args, answer, output_path, output_name):
    size = "large" if args.use_large else "small"
    file_name = os.path.join(output_path, '_'.join([args.data, size, output_name]))
    with open(file_name, 'w') as json_file:
        json.dump(answer, json_file, indent=2)
    print(f"JSON file '{file_name}' written.")

def load_predict_data(data_path, file_name):
    data_file = os.path.join(data_path, file_name)
    with open(data_file,'r') as f:
        data_dict = json.load(f)
    return data_dict

def describe_final_output(answer):
    for key in answer.keys():
        print(f"{key}: {len(answer[key])}")

def main(args): # given label classification
    print(args.use_large)
    start_time = time.time()
    client = ini_client(args.api_key)
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    label_list = get_predict_labels(args.output_path, args.data)
    print(f"Length of label list: {len(label_list)}")
    answer = known_label_categorize(args, client, data_list, label_list)
    answer = {k:v for k,v in answer.items() if len(v)!=0} # remove empty labels
    write_answer_to_json(args, answer, args.output_path, args.output_file_name)
    print(f"Classification result:")
    describe_final_output(answer)
    print(f"Total time usage: {time.time() - start_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--output_file_name", type=str, default="find_labels.json")
    parser.add_argument("--use_large", action="store_true", help="Use large model if set, otherwise use small model") # True - Large; False - Small
    parser.add_argument("--print_details", type=bool, default=False) # print details
    parser.add_argument("--test_num", type=int, default=5) # how many test numbers
    parser.add_argument("--api_key", type=str, default="sk-", help="set the key to your OpenAI Key")
    args = parser.parse_args()
    main(args)
