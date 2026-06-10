#!/bin/bash

echo "Scenario 2 / Hypothesis 2: Stealth dirty-label vs clean-label"
echo "This script runs BOTH experiments and creates txt logs in output_logs/"

# Stealth scenario:
# This is NOT the non-IID scenario.
# class_per_agent=10 means normal/IID-ish data.
# Compare dirty vs clean in terms of labels_changed, Val_Acc, Val_Loss, and Poison Acc.

python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_stealth_data=1 --verify_poisoning=1 --seed=1

python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --verify_stealth_data=1 --verify_poisoning=1 --seed=1

echo "Scenario 2 stealth experiments finished."
