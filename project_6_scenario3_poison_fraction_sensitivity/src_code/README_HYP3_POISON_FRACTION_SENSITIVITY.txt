README - Scenario 3 / Hypothesis 3: Poison Fraction Sensitivity
==============================================================

This project is Scenario 3 for the thesis.

Scenario:
---------
Poison fraction sensitivity.

Question:
---------
How does backdoor performance change when the attacker poisons fewer or more samples?

Poison fractions tested:
------------------------
0.05
0.10
0.25
0.50

This is more than just changing a command-line hyperparameter:
-------------------------------------------------------------
The code was actively changed for this scenario.

Main active code changes:
-------------------------

1. options.py
   Added:
   - --verify_poison_fraction_sensitivity
   - --poison_frac_grid
   - --scenario_name

2. utils.py
   Added:
   - get_poison_fraction_level(poison_frac)
   - VERIFY_POISON_FRACTION_SENSITIVITY_START / END log block
   - poison_fraction_level in poison verification
   - actual_poison_ratio in poison verification
   - POISON_FRACTION_SENSITIVITY_POINT log lines

3. federated.py
   Changed:
   - run name now starts with hyp3_poisonfrac_sensitivity
   - run name includes the exact poison fraction and poison level
   - txt log prints POISON_FRACTION_SENSITIVITY_RUN_START
   - every evaluation round prints the current poison fraction
   - TensorBoard logs Scenario3/Poison_Fraction

4. runner.sh
   Runs all eight thesis runs:
   - dirty-label pf=0.05
   - clean-label pf=0.05
   - dirty-label pf=0.10
   - clean-label pf=0.10
   - dirty-label pf=0.25
   - clean-label pf=0.25
   - dirty-label pf=0.50
   - clean-label pf=0.50

Data setting:
-------------
The data distribution is kept fixed:
--class_per_agent=10

This means Scenario 3 isolates poison-fraction sensitivity and does not mix it with non-IID effects.

Full commands:
--------------

Dirty-label pf=0.05:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.05 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1

Clean-label pf=0.05:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.05 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1

Dirty-label pf=0.10:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.10 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1

Clean-label pf=0.10:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.10 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1

Dirty-label pf=0.25:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.25 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1

Clean-label pf=0.25:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.25 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1

Dirty-label pf=0.50:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.50 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1

Clean-label pf=0.50:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.50 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1

What to compare:
----------------
From output_logs, compare:
- current_poison_frac
- poison_fraction_level
- poisoned_samples
- actual_poison_ratio
- labels_changed
- Val_Acc
- Val_Loss
- Poison Acc
- Poison Loss

Thesis interpretation:
----------------------
If poison accuracy rises as poison_frac increases, the backdoor attack is sensitive to poisoning intensity.
If validation accuracy falls as poison_frac increases, the attack becomes less stealthy at higher poisoning rates.
Clean-label should keep labels_changed=0 at every poison fraction.
Dirty-label should show labels_changed=poisoned_samples at every poison fraction.
