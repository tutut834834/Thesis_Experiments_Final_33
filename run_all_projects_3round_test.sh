#!/usr/bin/env bash
# run_all_projects_3round_test.sh
#
# Master smoke-test runner for all thesis projects so far.
# Purpose:
#   - Run each experiment for ONLY 3 rounds.
#   - Check whether the project starts correctly.
#   - Check whether output_logs/*.txt files are created.
#
# Use in Git Bash:
#   bash run_all_projects_3round_test.sh
#
# IMPORTANT:
# This script is based on your extracted GitHub folder:
#   C:\Users\Alexa\Downloads\Thesis_Experiments_Final_33-main\Thesis_Experiments_Final_33-main

set +e

PROJECT_ROOT="/c/Users/Alexa/Downloads/Thesis_Experiments_Final_33-main/Thesis_Experiments_Final_33-main"
MASTER_LOG_DIR="$PROJECT_ROOT/master_test_logs"
MASTER_LOG="$MASTER_LOG_DIR/master_3round_test_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$MASTER_LOG_DIR"

echo "============================================================" | tee -a "$MASTER_LOG"
echo "MASTER 3-ROUND TEST RUNNER" | tee -a "$MASTER_LOG"
echo "PROJECT_ROOT=$PROJECT_ROOT" | tee -a "$MASTER_LOG"
echo "START=$(date)" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

run_exp () {
    local PROJECT_NAME="$1"
    local EXP_NAME="$2"
    local CMD="$3"
    local SRC="$PROJECT_ROOT/$PROJECT_NAME/src_code"

    echo "" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"
    echo "PROJECT: $PROJECT_NAME" | tee -a "$MASTER_LOG"
    echo "EXPERIMENT: $EXP_NAME" | tee -a "$MASTER_LOG"
    echo "SRC: $SRC" | tee -a "$MASTER_LOG"
    echo "CMD: $CMD" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"

    if [ ! -d "$SRC" ]; then
        echo "SKIP: src_code folder not found: $SRC" | tee -a "$MASTER_LOG"
        return 0
    fi

    cd "$SRC" || {
        echo "SKIP: could not cd into $SRC" | tee -a "$MASTER_LOG"
        return 0
    }

    mkdir -p output_logs

    BEFORE_COUNT=$(find output_logs -maxdepth 1 -type f -name "*.txt" 2>/dev/null | wc -l)

    echo "Before txt count in output_logs: $BEFORE_COUNT" | tee -a "$MASTER_LOG"
    echo "Running now..." | tee -a "$MASTER_LOG"

    bash -lc "$CMD"
    EXIT_CODE=$?

    AFTER_COUNT=$(find output_logs -maxdepth 1 -type f -name "*.txt" 2>/dev/null | wc -l)

    echo "Exit code: $EXIT_CODE" | tee -a "$MASTER_LOG"
    echo "After txt count in output_logs: $AFTER_COUNT" | tee -a "$MASTER_LOG"

    if [ "$AFTER_COUNT" -gt "$BEFORE_COUNT" ]; then
        echo "TXT_CREATED: YES" | tee -a "$MASTER_LOG"
        echo "Newest txt file:" | tee -a "$MASTER_LOG"
        ls -t output_logs/*.txt 2>/dev/null | head -n 1 | tee -a "$MASTER_LOG"
    else
        echo "TXT_CREATED: NO or same count" | tee -a "$MASTER_LOG"
    fi

    echo "Finished: $PROJECT_NAME / $EXP_NAME" | tee -a "$MASTER_LOG"
}

# ============================================================
# Base / earlier projects
# ============================================================

run_exp "project_-2_clean_baseline" "clean_baseline_3round" \
"python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3"

run_exp "project_-1_dirty_attack_baseline_paper_learning_rate" "dirty_attack_baseline_3round" \
"python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --num_corrupt=1 --poison_frac=0.5"

run_exp "project_0_learning_rate_command3" "robustlr_learning_rate_3round" \
"python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --num_corrupt=1 --poison_frac=0.5 --robustLR_threshold=4"

# ============================================================
# Three clean-label projects before the scenarios
# ============================================================

run_exp "project_1_cleanlabelplus" "cleanlabel_plus_3round" \
"python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --clean_label=1 --clean_label_type=1"

run_exp "project_2_cleanlabelsquare" "cleanlabel_square_3round" \
"python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --clean_label=1 --clean_label_type=2"

run_exp "project_3_cleanlabelapple" "cleanlabel_apple_3round" \
"python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --clean_label=1 --clean_label_type=3"

# ============================================================
# Project 4 / Scenario 1: non-IID dirty vs clean
# Important: class_per_agent=2 means corrupt client has classes 0 and 1.
# ============================================================

run_exp "project_4_scenario1_noniid_dirty_vs_clean" "hyp1_noniid_dirty_3round" \
"python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=2 --base_class=0 --target_class=1 --clean_label=0 --verify_noniid=1 --verify_poisoning=1 --seed=1"

run_exp "project_4_scenario1_noniid_dirty_vs_clean" "hyp1_noniid_clean_3round" \
"python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=2 --base_class=0 --target_class=1 --clean_label=1 --clean_label_type=1 --verify_noniid=1 --verify_poisoning=1 --seed=1"

# ============================================================
# Project 5 / Scenario 2: low-poison / stealth dirty vs clean
# ============================================================

run_exp "project_5_scenario2_lowpoison_stealth" "hyp2_lowpoison_dirty_3round" \
"python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.1 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_stealth_data=1 --verify_poisoning=1 --seed=1"

run_exp "project_5_scenario2_lowpoison_stealth" "hyp2_lowpoison_clean_3round" \
"python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.1 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --verify_stealth_data=1 --verify_poisoning=1 --seed=1"

# ============================================================
# Project 6 / Scenario 3: poison fraction sensitivity
# ============================================================

for PF in 0.05 0.10 0.25 0.50
do
    PF_TAG=$(echo "$PF" | sed 's/\.//g')

    run_exp "project_6_scenario3_poison_fraction_sensitivity" "hyp3_pf${PF_TAG}_dirty_3round" \
    "python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=$PF --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1"

    run_exp "project_6_scenario3_poison_fraction_sensitivity" "hyp3_pf${PF_TAG}_clean_3round" \
    "python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=$PF --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --verify_poison_fraction_sensitivity=1 --verify_poisoning=1 --seed=1"
done

# ============================================================
# Project 7 / Scenario 4: defense comparison
# ============================================================

for DEF in none robustlr comed clip_noise
do
    run_exp "project_7_scenario4_defense_comparison" "hyp4_${DEF}_dirty_3round" \
    "python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --defense_mode=$DEF --verify_defense_comparison=1 --verify_poisoning=1 --seed=1"

    run_exp "project_7_scenario4_defense_comparison" "hyp4_${DEF}_clean_3round" \
    "python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --defense_mode=$DEF --verify_defense_comparison=1 --verify_poisoning=1 --seed=1"
done

# ============================================================
# Project 8 / Scenario 5: trigger geometry
# ============================================================

for TRIG in plus square apple
do
    if [ "$TRIG" = "plus" ]; then CLTYPE=1; fi
    if [ "$TRIG" = "square" ]; then CLTYPE=2; fi
    if [ "$TRIG" = "apple" ]; then CLTYPE=3; fi

    run_exp "project_8_scenario5_trigger_geometry" "hyp5_${TRIG}_dirty_3round" \
    "python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --trigger_geometry=$TRIG --pattern_type=$TRIG --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1"

    run_exp "project_8_scenario5_trigger_geometry" "hyp5_${TRIG}_clean_3round" \
    "python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=$CLTYPE --trigger_geometry=$TRIG --pattern_type=$TRIG --verify_trigger_geometry=1 --verify_poisoning=1 --seed=1"
done

# ============================================================
# Project 9 / Scenario 6: privacy-preserving
# ============================================================

for PRIV in none low medium high
do
    run_exp "project_9_scenario6_privacy_preserving" "hyp6_privacy_${PRIV}_dirty_3round" \
    "python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=0 --privacy_mode=clip_noise --privacy_level=$PRIV --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1"

    run_exp "project_9_scenario6_privacy_preserving" "hyp6_privacy_${PRIV}_clean_3round" \
    "python -u federated.py --data=fmnist --local_ep=2 --bs=256 --num_agents=10 --rounds=3 --snap=1 --num_corrupt=1 --poison_frac=0.5 --class_per_agent=10 --base_class=5 --target_class=7 --clean_label=1 --clean_label_type=1 --privacy_mode=clip_noise --privacy_level=$PRIV --verify_privacy_preserving=1 --verify_poisoning=1 --seed=1"
done

echo "" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "ALL 3-ROUND TESTS FINISHED" | tee -a "$MASTER_LOG"
echo "END=$(date)" | tee -a "$MASTER_LOG"
echo "MASTER_LOG=$MASTER_LOG" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"

echo ""
echo "Done. Master log:"
echo "$MASTER_LOG"
