#!/bin/bash

echo "Scenario 5 / Hypothesis 5: Trigger geometry"
echo "Runs dirty-label and clean-label attacks for plus, square, apple triggers."

for TRIG in plus square apple
do
  echo "Running DIRTY-LABEL trigger geometry ${TRIG}"
  python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --trigger_geometry=${TRIG} --pattern_type=${TRIG} --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1

  if [ "$TRIG" = "plus" ]; then CLTYPE=1; fi
  if [ "$TRIG" = "square" ]; then CLTYPE=2; fi
  if [ "$TRIG" = "apple" ]; then CLTYPE=3; fi

  echo "Running CLEAN-LABEL trigger geometry ${TRIG}"
  python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=${CLTYPE} --trigger_geometry=${TRIG} --pattern_type=${TRIG} --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1
done

echo "Scenario 5 trigger geometry experiments finished."
