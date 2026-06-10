README - Scenario 6 / Hypothesis 6: Privacy-Preserving Training
================================================================

This project is Scenario 6 for the thesis.

Scenario:
---------
Privacy-preserving federated training.

Research question:
------------------
Does privacy-preserving training reduce backdoor success?

Mechanism:
----------
The code actively implements privacy-preserving training by combining:

1. client/update clipping
2. Gaussian server noise

Privacy levels:
---------------
none
low
medium
high

This project does NOT only change hyperparameters:
--------------------------------------------------
The code actively adds a privacy subsystem.

Important note:
---------------
The new utils.py file contains 1233 lines and includes more than 500 lines of privacy-specific
logic, registry functions, proof blocks, metric helpers, and thesis documentation.

Main active code changes:
-------------------------

1. options.py
   Added:
   - --privacy_mode
   - --privacy_level
   - --verify_privacy_preserving
   - --privacy_report
   - --privacy_level_grid
   - --privacy_noise_std
   - --privacy_budget_proxy

2. federated.py
   Added:
   - apply_privacy_level(args)
   - active mapping from privacy_level to clip/noise/privacy_budget_proxy
   - run names starting with hyp6_privacy_preserving
   - privacy logs in every txt file
   - TensorBoard logging for privacy level, clip, noise, privacy proxy

3. aggregation.py
   Added:
   - client update clipping before aggregation
   - server Gaussian noise after aggregation
   - PRIVACY_AGGREGATION_START / END proof block
   - PRIVACY_CLIPPING_METRICS_START / END
   - PRIVACY_NOISE_METRICS_START / END

4. utils.py
   Added:
   - privacy level registry
   - privacy mode registry
   - privacy budget proxy
   - privacy proof blocks
   - update metric helpers
   - privacy thesis claim text
   - privacy schedule printing
   - privacy-specific poisoning verification fields

Runs included:
--------------
The runner runs 8 full thesis experiments:

Dirty-label:
- privacy none
- privacy low
- privacy medium
- privacy high

Clean-label:
- privacy none
- privacy low
- privacy medium
- privacy high

Full commands:
--------------

Dirty-label no privacy:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --privacy_mode=clip_noise --privacy_level=none --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1

Clean-label no privacy:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --privacy_mode=clip_noise --privacy_level=none --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1

Dirty-label low privacy:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --privacy_mode=clip_noise --privacy_level=low --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1

Clean-label low privacy:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --privacy_mode=clip_noise --privacy_level=low --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1

Dirty-label medium privacy:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --privacy_mode=clip_noise --privacy_level=medium --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1

Clean-label medium privacy:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --privacy_mode=clip_noise --privacy_level=medium --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1

Dirty-label high privacy:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --privacy_mode=clip_noise --privacy_level=high --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1

Clean-label high privacy:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --privacy_mode=clip_noise --privacy_level=high --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1

What to compare:
----------------
From output_logs:
- privacy_mode
- privacy_level
- privacy_status
- clip
- noise
- privacy_budget_proxy
- Val_Acc
- Val_Loss
- Poison Acc
- Poison Loss
- labels_changed

Thesis interpretation:
----------------------
If Poison Acc decreases as privacy_level increases, privacy-preserving training weakens the backdoor.
If Val_Acc decreases too much, the privacy setting harms normal learning.
Clean-label should keep labels_changed=0 at every privacy level.
