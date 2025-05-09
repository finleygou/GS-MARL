#!/bin/bash

# Run the script
seed_max=1
ep_lens=200

for seed in `seq ${seed_max}`;
do
# seed=`expr ${seed} + 1`
echo "seed: ${seed}"
# execute the script with different params
python  ../gsmarl/scripts/train_mpe.py \
--device "cuda:3" \
--wandb_project_name "Graph_Safe_MARL" \
--env_name "MPE" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "simple_encirclement_3agt_tune" \
--max_edge_dist 2.0 \
--safety_bound 1.0 \
--clip_param 0.15 --gamma 0.99 \
--hidden_size 64 \
--num_targets 1 --num_agents 3 --num_obstacles 5 --num_landmarks 0 \
--save_data "False" \
--reward_file_name "rew_enc_3agt_PS-v15" \
--cost_file_name "cost_enc_3agt_PS-v15" \
--kl_threshold 0.1 \
--parameter_share "True" \
--use_curriculum "True" --cp 0.6 \
--use_wandb "True" \
--n_training_threads 128 --n_rollout_threads 32 \
--episode_length ${ep_lens} \
--num_env_steps 6000000 \
--data_chunk_length 15 \
--gain 0.01 --lr 5e-4 --critic_lr 5e-4 \
--model_dir "/data/goufandi_space/Projects/GS-MARL/gsmarl/results/MPE/simple_encirclement_3agt/gsmarl/check/wandb/run-20250509_020855-77gnes5i/files/" \
--user_name "finleygou"
done