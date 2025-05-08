#!/bin/bash
set -e
# Run the script
seed_max=1
ep_lens=150

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
# execute the script with different params
python  ../gsmarl/scripts/render_mpe.py \
--device "cpu" \
--use_valuenorm --use_popart \
--env_name "MPE" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "simple_encirclement_3agt" \
--max_edge_dist 2.0 \
--hidden_size 64 \
--num_targets 1 --num_agents 3 --num_obstacles 5 --num_landmarks 0 \
--save_data "False" \
--save_gifs "False" \
--use_render "True" \
--parameter_share "True" \
--episode_length ${ep_lens} \
--data_chunk_length 20 \
--gain 0.01 \
--render_episodes 5 \
--user_name "finleygou" \
--model_dir "/data/goufandi_space/Projects/GS-MARL/gsmarl/results/MPE/simple_encirclement_3agt/gsmarl/check/wandb/run-20250429_222546-kwz6hrx6/files/"
done 