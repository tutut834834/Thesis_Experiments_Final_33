@echo off
echo Running H4 defense comparison DIRTY-LABEL defense=none, 200 rounds...
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --defense_mode=none --verify_defense_comparison=1 --verify_poisoning=1 --seed=1
pause
