@echo off
echo Running H6 privacy-preserving CLEAN-LABEL privacy=none, 200 rounds...
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --privacy_mode=clip_noise --privacy_level=none --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1
pause
