path="generated_labels"
# Label generation
for dataset in arxiv_fine go_emotion massive_intent massive_scenario mtop_intent
do
    nohup python label_generation.py \
        --data $dataset \
    > ./logs/${path}/${dataset}_small_label_generation.log 2>&1 &
done

wait
# Given labels classification
for dataset in arxiv_fine go_emotion massive_intent massive_scenario mtop_intent
do
    nohup python given_label_classification.py \
        --data $dataset \
    > ./logs/${path}/${dataset}_small_given_label_generation.log 2>&1 &
done

wait
# Evaluation
for dataset in arxiv_fine go_emotion massive_intent massive_scenario mtop_intent
do
    nohup python evaluate.py \
    --data $dataset \
    --predict_file_path ./${path}/ \
    --predict_file ${dataset}_small_find_labels.json \
    > ./logs/${path}/evaluate_${dataset}.log 2>&1 &
done