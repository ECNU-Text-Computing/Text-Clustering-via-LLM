# Text Clustering as Classification with LLMs

This repository contains the implementation code for the paper "[Text Clustering as Classification with LLMs](https://arxiv.org/abs/2410.00927)".

## Initial Setup

To run the code, you need to set the `--api_key` parameter to your own OpenAI Key. Please note that OpenAI may have updated their API since the last time this code was run.

## Experiments

### Step 1: Download the Dataset

First, download the dataset from the following link: [Dataset](https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view?usp=sharing), as provided by the paper [CLUSTERLLM: Large Language Models as a Guide for Text Clustering (EMNLP2023)](https://aclanthology.org/2023.emnlp-main.858/).

### Step 2: Select Part Labels

Run the following command to select the labels that will be shown to the LLM:
```bash
python select_part_labels.py
```
The selected labels will be saved in `chosen_labels.json` within the `generated_labels` folder.

### Step 3: Run the Code
Execute the script to perform the clustering process:
```bash
bash run.sh
```
This script will run the following files:
- `label_generation.py`: Generates potential labels.
- `given_label_classification.py`: Classifies the data based on the generated labels.
- `evaluate.py`: Evaluates the final clustering results.

The code will generate the following files:
- `{dataset}_llm_generated_labels_before_merge.json`
- `{dataset}_llm_generated_labels_after_merge.json`
- `{dataset}_find_labels.json`

