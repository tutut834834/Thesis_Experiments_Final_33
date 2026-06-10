README - Scenario 5 / Hypothesis 5: Trigger Geometry
====================================================

This project is Scenario 5 for the thesis.

Scenario:
---------
Trigger geometry comparison.

Trigger geometries:
-------------------
1. plus
2. square
3. apple

Research question:
------------------
Does trigger shape/design influence backdoor success?

Why this scenario matters:
--------------------------
If hand-designed trigger geometries lead to different Poison Acc values, then trigger design matters.
This prepares the later Cuckoo Search extension, because Cuckoo Search can be used to optimize trigger
shape/location/pixels automatically.

This project does NOT only change a hyperparameter:
--------------------------------------------------
The code actively adds a trigger geometry subsystem.

Important note:
---------------
The utils.py file contains 1277 lines. The project includes more than 500 lines of added
trigger-geometry logic, metrics, proof logs, registry functions, mask builders, and documentation.

Main active code changes:
-------------------------

1. options.py
   Added:
   - --trigger_geometry
   - --trigger_geometry_grid
   - --verify_trigger_geometry
   - --trigger_geometry_report
   - --scenario_name

2. utils.py
   Added many trigger geometry functions:
   - normalize_trigger_geometry
   - clean_label_type_to_geometry
   - get_geometry_from_args
   - geometry_to_clean_label_type
   - get_geometry_family
   - get_geometry_expected_visibility
   - build_plus_mask
   - build_square_mask
   - build_apple_proxy_mask
   - build_trigger_mask
   - compute_trigger_geometry_metrics
   - print_trigger_geometry_metrics
   - print_geometry_grid_comparison
   - print_trigger_registry
   - all_trigger_geometry_setup_logs
   - trigger_geometry_summary_line
   - many additional proof/metric helpers

3. federated.py
   Changed:
   - run names start with hyp5_trigger_geometry
   - run names include trigplus/trigsquare/trigapple
   - txt logs print TRIGGER_GEOMETRY_RUN_START
   - TensorBoard logs trigger geometry index, mask area, density, compactness
   - every evaluation round prints trigger geometry summary

4. runner.sh
   Runs all 6 full thesis experiments:
   - dirty-label plus
   - clean-label plus
   - dirty-label square
   - clean-label square
   - dirty-label apple
   - clean-label apple

Full commands:
--------------

Dirty-label plus:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --trigger_geometry=plus --pattern_type=plus --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1

Clean-label plus:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --trigger_geometry=plus --pattern_type=plus --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1

Dirty-label square:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --trigger_geometry=square --pattern_type=square --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1

Clean-label square:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=2 --trigger_geometry=square --pattern_type=square --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1

Dirty-label apple:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --trigger_geometry=apple --pattern_type=apple --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1

Clean-label apple:
python federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=200 --snap=10 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=3 --trigger_geometry=apple --pattern_type=apple --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1

What to compare:
----------------
From output_logs:
- trigger_geometry
- trigger_geometry_family
- trigger_mask_area
- trigger_mask_density
- trigger_compactness_score
- changed_pixels_first_sample
- labels_changed
- Val_Acc
- Val_Loss
- Poison Acc
- Poison Loss

Thesis interpretation:
----------------------
If Poison Acc differs across plus, square, and apple, then trigger geometry matters.
If trigger geometry matters, automatic trigger optimization is justified.
This motivates the later Cuckoo Search scenario.
