#!/bin/bash

echo "Scenario 6 / Hypothesis 6: Privacy-preserving training"
echo "Runs dirty-label and clean-label attacks for privacy levels none, low, medium, high."

for PRIV in none low medium high
do
  echo "Running DIRTY-LABEL privacy level ${PRIV}"
  python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --privacy_mode=clip_noise --privacy_level=${PRIV} --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1

  echo "Running CLEAN-LABEL privacy level ${PRIV}"
  python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --privacy_mode=clip_noise --privacy_level=${PRIV} --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1
done

echo "Scenario 6 privacy-preserving experiments finished."
