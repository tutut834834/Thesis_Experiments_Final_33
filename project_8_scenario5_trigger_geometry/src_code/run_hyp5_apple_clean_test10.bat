@echo off
echo Running H5 trigger geometry CLEAN-LABEL trigger=apple, 10-round test...
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=10 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=3 --trigger_geometry=apple --pattern_type=apple --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1
pause
