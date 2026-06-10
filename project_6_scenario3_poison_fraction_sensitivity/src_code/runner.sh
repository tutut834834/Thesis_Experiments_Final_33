#!/bin/bash

echo "Scenario 3 / Hypothesis 3: Poison fraction sensitivity"
echo "Runs dirty-label and clean-label for poison_frac = 0.05, 0.10, 0.25, 0.50"
echo "Each run creates a separate txt log in output_logs/"

for PF in 0.05 0.10 0.25 0.50
do
  echo "Running DIRTY-LABEL poison fraction ${PF}"
  python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=${PF} --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1

  echo "Running CLEAN-LABEL poison fraction ${PF}"
  python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=${PF} --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1
done

echo "Scenario 3 poison fraction sensitivity experiments finished."
