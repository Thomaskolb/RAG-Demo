SIZE=base
DATA_DIR=./atlas_data
MODEL_PATH=/home/tkolb/data/models/atlas/base
INDEX_PATH=/home/tkolb/data/indices/atlas/wiki/base
EVAL_PATH=/home/tkolb/data/datasets/queries/qs.json
EXPERIMENT_NAME=my-nq-64-shot-example
port=$(shuf -i 15000-16000 -n 1)

source /home/tkolb/data/envs/rag/bin/activate

python /home/tkolb/RAG-Demo/atlas/evaluate.py \
    --name 'my-nq-64-shot-example-evaluation' \
    --generation_max_length 16 \
    --gold_score_mode "pdist" \
    --precision fp32 \
    --reader_model_type google/t5-${SIZE}-lm-adapt \
    --text_maxlength 512 \
    --model_path ${MODEL_PATH} \
    --eval_data ${EVAL_PATH} \
    --per_gpu_batch_size 1 \
    --n_context 40 --retriever_n_context 40 \
    --checkpoint_dir ${MODEL_PATH} \
    --main_port $port \
    --index_mode "flat"  \
    --task "qa" \
    --load_index_path ${INDEX_PATH} \
    --write_results

deactivate