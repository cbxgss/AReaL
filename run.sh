set_variable_with_default() {
    local var_name=$1
    shift
    local guidance=$1
    shift
    local default_values=("$@")
    local num_options=${#default_values[@]}

    # 打印所有选项
    for ((i = 0; i < num_options; i++)); do
        echo "$((i + 1)). ${default_values[$i]}"
    done

    read -p "input $guidance (input 1-$num_options to choose a default value): " user_input

    if [[ $user_input =~ ^[1-$num_options]$ ]]; then
        local selected_index=$((user_input - 1))
        export "$var_name=${default_values[$selected_index]}"
    else
        export "$var_name=$user_input"
    fi

    echo "Selected $guidance: ${var_name} = ${!var_name}"
    echo ""
}

set_variable_with_default CUDA_VISIBLE_DEVICES device 0,1 2,3 4,5 6,7 0,1,2,3 4,5,6,7 0,1,2,3,4,5,6,7
gpu_nums=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

experiment_name="async_ppo_qwen2.5"

current_dir=$(pwd)

mkdir -p ${current_dir}/outputs/$experiment_name

python training/main_async_ppo.py \
    n_nodes=1 n_gpus_per_node=${gpu_nums} \
    allocation_mode=sglang.d4p1m1+d2p2m1 \
    cluster.fileroot=${current_dir}/outputs/$experiment_name \
    actor.type._class=qwen2 \
    actor.path=Qwen/Qwen2.5-3B-Instruct \
    ref.type._class=qwen2.5 \
    ref.path=Qwen/Qwen2.5-3B-Instruct \
    dataset.path=$HF_HOME/hub/datasets--inclusionAI--AReaL-RL-Data/snapshots/07dfc8977909d38366d4e913eb3648a939e5aeb4/data/deepscaler_40k_0319.jsonl \
    dataset.train_bs_n_seqs=32 \
    group_size=8 \
    ppo.gen.max_new_tokens=8196 \
    ppo.ppo_n_minibatches=4 \
    actor.sglang.attention_backend=fa3 \
    actor_train.mb_spec.max_tokens_per_mb=32768 \
    actor_inf.mb_spec.max_tokens_per_mb=32768 \
    max_concurrent_rollouts=16 \
    max_head_offpolicyness=4 \
    mem_per_master_worker=2000 mem_per_model_worker=9000 \
    2>&1 | tee ${current_dir}/outputs/$experiment_name/log.log
