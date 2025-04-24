#!/bin/bash

# Run the script
seed_max=1
ep_lens=150

for seed in `seq ${seed_max}`;
do
# seed=`expr ${seed} + 1`
echo "seed: ${seed}"
# execute the script with different params
python  ../gsmarl/scripts/train_mpe.py \
--device "cuda:1" \
--use_valuenorm --use_popart \
--wandb_project_name "Graph_Safe_MARL" \
--env_name "MPE" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "simple_encirclement_5agt" \
--max_edge_dist 2.0 \
--safety_bound 1.0 \
--clip_param 0.15 --gamma 0.99 \
--hidden_size 64 \
--num_targets 1 --num_agents 5 --num_obstacles 5 --num_landmarks 0 \
--save_data "True" \
--reward_file_name "rew_enc_5agt_PS" \
--cost_file_name "cost_enc_5agt_PS" \
--parameter_share "True" \
--use_wandb "True" \
--n_training_threads 128 --n_rollout_threads 16 \
--episode_length ${ep_lens} \
--num_env_steps 6000000 \
--data_chunk_length 20 \
--gain 0.01 --lr 2e-4 --critic_lr 2e-4 \
--user_name "finleygou"
done