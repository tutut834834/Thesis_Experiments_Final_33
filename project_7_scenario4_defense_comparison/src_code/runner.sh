#!/bin/bash

echo "Scenario 4 / Hypothesis 4: Defense comparison"
echo "Runs dirty-label and clean-label attacks under multiple defenses."
echo "Defenses: none, robustlr, coordinate median, clipping+noise"

for DEF in none robustlr comed clip_noise
do
  echo "Running DIRTY-LABEL defense ${DEF}"
  python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --defense_mode=${DEF} --verify_defense_comparison=1 --verify_poisoning=1 --seed=1

  echo "Running CLEAN-LABEL defense ${DEF}"
  python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --defense_mode=${DEF} --verify_defense_comparison=1 --verify_poisoning=1 --seed=1
done

echo "Scenario 4 defense comparison experiments finished."
