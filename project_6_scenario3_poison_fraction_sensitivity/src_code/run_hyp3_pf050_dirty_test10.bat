@echo off
echo Running H3 poison fraction sensitivity DIRTY-LABEL pf=0.50, 10-round test...
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=10 --snap=1 --num_corrupt=1 --poison_frac=0.50 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1
pause
