#!/bin/bash

# Run the script
seed_max=1
n_agents=3
# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")
ep_lens=100

for seed in `seq ${seed_max}`;
do
# seed=`expr ${seed} + 1`
echo "seed: ${seed}"
# execute the script with different params
python  ../gsmarl/scripts/train_mpe.py \
--device "cuda:0" \
--use_valuenorm --use_popart \
--project_name "Graph_Safe_MARL" \
--env_name "MPE" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "exp2" \
--max_edge_dist 0.8 \
--clip_param 0.15 --gamma 0.99 \
--hidden_size 64 \
--save_data "True" \
--reward_file_name "rew1" \
--cost_file_name "cost1" \
--use_wandb "False" \
--n_training_threads 8 --n_rollout_threads 16 \
--episode_length ${ep_lens} \
--num_env_steps 10000000 \
--data_chunk_length 20 \
--gain 0.01 --lr 2e-4 --critic_lr 2e-4 \
--user_name "finleygou" 
done