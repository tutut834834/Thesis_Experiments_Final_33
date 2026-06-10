README - Scenario 2 / Hypothesis 2: Stealth
==========================================

This project is derived from the uploaded Scenario 1 non-IID code, but changed for the specific
Scenario 2 stealth experiment.

Core idea:
----------
Scenario 1 tested non-IID data heterogeneity.
Scenario 2 tests stealth.

Therefore, Scenario 2 does NOT use the non-IID data setting.
It uses:

    --class_per_agent=10

This is the normal/IID-ish federated setting. The goal is to compare dirty-label and clean-label
attacks under the same ordinary data distribution.

Hypothesis 2:
-------------
Clean-label attacks are more stealthy than dirty-label attacks because clean-label poisoned
training samples keep their original labels.

Expected verification:
----------------------
Dirty-label run:
    attack_mode=dirty-label
    labels_changed=poisoned_samples
    STEALTH_COMPARISON_POINT

Clean-label run:
    attack_mode=clean-label-1
    labels_changed=0
    STEALTH_PROOF

Main files changed from the non-IID version:
-------------------------------------------
1. options.py
   - Renamed the scenario options around stealth.
   - Added:
        --verify_stealth_data
   - Kept:
        --class_per_agent
        --seed
        --clean_label
        --clean_label_type
        --verify_poisoning

2. utils.py
   - distribute_data is now documented as the stealth data setting.
   - The verification block is renamed:
        VERIFY_STEALTH_DATA_START
        VERIFY_STEALTH_DATA_PASSED
        VERIFY_STEALTH_DATA_END
   - It prints:
        scenario_data=IID-ish / normal federated data
   - poison_dataset now prints:
        STEALTH_PROOF for clean-label
        STEALTH_COMPARISON_POINT for dirty-label

3. federated.py
   - run_name changed from:
        hyp1_noniid
     to:
        hyp2_stealth
   - Keeps full txt logging in output_logs/.
   - Adds:
        STEALTH_METRICS_NOTE

4. runner.sh
   - Runs dirty-label and clean-label full 200-round stealth experiments.
   - Uses:
        --class_per_agent=10
        --base_class=5
        --target_class=7

Quick commands:
---------------

Dirty-label quick test:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=10 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_stealth_data=1 --verify_poisoning=1 --seed=1

Clean-label quick test:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=10 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --verify_stealth_data=1 --verify_poisoning=1 --seed=1

Full commands:
--------------

Dirty-label full:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_stealth_data=1 --verify_poisoning=1 --seed=1

Clean-label full:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --verify_stealth_data=1 --verify_poisoning=1 --seed=1

What to compare in the txt files:
---------------------------------
From output_logs, compare:
- labels_changed
- Val_Loss
- Val_Acc
- Val_Per_Class_Acc
- Poison Loss
- Poison Acc
- changed_pixels_first_sample

Thesis interpretation:
----------------------
Dirty-label is less stealthy because it visibly changes labels in the training data.
Clean-label is more stealthy because labels_changed=0.
If Val_Acc and Val_Loss stay close to the dirty-label/baseline run, that supports the stealth argument.
