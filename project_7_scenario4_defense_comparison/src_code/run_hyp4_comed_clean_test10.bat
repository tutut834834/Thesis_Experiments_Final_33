@echo off
echo Running H4 defense comparison CLEAN-LABEL defense=comed, 10-round test...
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=10 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --defense_mode=comed --verify_defense_comparison=1 --verify_poisoning=1 --seed=1
pause
