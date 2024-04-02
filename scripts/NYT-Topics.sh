data_name='NYT-Topics'
model_name='princeton-nlp/sup-simcse-roberta-base'
device='cuda:3'
data_dir=datasets/$data_name
cache_dir=cache
output_dir=output
mkdir -p $output_dir

python src/rule_prompt.py \
    --dataset $data_name \
    --device $device \
    --data_dir $data_dir \
    --cache_dir $cache_dir \
    --simcse_model_name_or_path $model_name \
    --num_iterations 5 \
    --select 8 \
    --support 0.05 \
    --max_len 150 \
    --learning_rate 1e-9 \
    --proportion_ft 0.9 \
    --batch_s 32 \


