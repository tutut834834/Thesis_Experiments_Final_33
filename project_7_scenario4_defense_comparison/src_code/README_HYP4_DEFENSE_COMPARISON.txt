README - Scenario 4 / Hypothesis 4: Defense Comparison
=====================================================

This project is Scenario 4 for the thesis.

Scenario:
---------
Defense comparison against dirty-label and clean-label backdoor attacks.

This project does NOT only change one hyperparameter.
The code actively implements a defense_mode system.

Defense modes:
--------------
1. none
   - FedAvg baseline
   - aggr=avg
   - robustLR_threshold=0
   - clip=0
   - noise=0

2. robustlr
   - Robust Learning Rate defense
   - aggr=avg
   - robustLR_threshold=4

3. comed
   - Coordinate-wise median aggregation defense
   - aggr=comed

4. clip_noise
   - Update clipping plus server noise
   - aggr=avg
   - clip=5.0
   - noise=0.01

Main active code changes:
-------------------------

1. options.py
   Added:
   - --defense_mode
   - --verify_defense_comparison
   - --scenario_name

2. federated.py
   Added:
   - apply_defense_mode(args)
   - automatic mapping from defense_mode to actual defense settings
   - run names beginning with hyp4_defense_comparison
   - DEFENSE_COMPARISON_RUN_START log block
   - defense metrics printed every evaluation round

3. aggregation.py
   Added:
   - get_defense_status()
   - print_defense_once()
   - AGGREGATION_DEFENSE_START / END proof block
   - active proof that robustlr / comed / clip_noise is really enabled

4. utils.py
   Added:
   - get_defense_status(args)
   - VERIFY_DEFENSE_COMPARISON_START / END proof block
   - defense_mode and defense_status inside poisoning verification

Runs included:
--------------
The runner runs 8 full thesis experiments:

Dirty-label:
- none
- robustlr
- comed
- clip_noise

Clean-label:
- none
- robustlr
- comed
- clip_noise

Full commands:
--------------

Dirty-label no defense:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --defense_mode=none --verify_defense_comparison=1 --verify_poisoning=1 --seed=1

Clean-label no defense:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --defense_mode=none --verify_defense_comparison=1 --verify_poisoning=1 --seed=1

Dirty-label RobustLR:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --defense_mode=robustlr --verify_defense_comparison=1 --verify_poisoning=1 --seed=1

Clean-label RobustLR:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --defense_mode=robustlr --verify_defense_comparison=1 --verify_poisoning=1 --seed=1

Dirty-label coordinate median:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --defense_mode=comed --verify_defense_comparison=1 --verify_poisoning=1 --seed=1

Clean-label coordinate median:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --defense_mode=comed --verify_defense_comparison=1 --verify_poisoning=1 --seed=1

Dirty-label clipping+noise:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --defense_mode=clip_noise --verify_defense_comparison=1 --verify_poisoning=1 --seed=1

Clean-label clipping+noise:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --defense_mode=clip_noise --verify_defense_comparison=1 --verify_poisoning=1 --seed=1

What to compare:
----------------
From output_logs:
- defense_mode
- defense_status
- aggr
- robustLR_threshold
- clip
- noise
- Val_Acc
- Val_Loss
- Poison Acc
- Poison Loss
- labels_changed

Thesis interpretation:
----------------------
If a defense works, Poison Acc should decrease compared with defense_mode=none.
If Val_Acc remains close to the no-defense setting, the defense suppresses the backdoor without destroying normal accuracy.
Dirty-label and clean-label can be compared under every defense.
