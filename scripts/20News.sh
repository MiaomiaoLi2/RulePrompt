data_name='20News'
model_name=princeton-nlp/sup-simcse-roberta-base
device='cuda:0'
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
    --select 7 \
    --max_len 500 \
    --epochs 30 \
    --batch_s 16 \
    --num_verbalizers 1 \

