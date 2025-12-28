#!/bin/bash
# =============================================================================
# MRI-GUARDIAN Complete Training Pipeline
# =============================================================================
# Trains all modalities on 4x A100 GPUs
#
# Usage:
#   chmod +x train_all.sh
#   ./train_all.sh
#
# This trains:
#   1. MRI (FastMRI) - ~28 hours
#   2. CT (LIDC + COVID) - ~20 hours
#   3. X-ray (ChestX-ray14) - ~18 hours
#   4. Runs all experiments
#   5. Packages results for download
# =============================================================================

set -e

# Configuration
DATA_DIR="/data/medical_imaging"
PROJECT_DIR="/home/ubuntu/mri-guardian"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
RESULTS_DIR="$PROJECT_DIR/results"
NUM_GPUS=4

# Logging
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p $LOG_DIR $CHECKPOINT_DIR $RESULTS_DIR

echo "============================================================"
echo "MRI-GUARDIAN Complete Training Pipeline"
echo "============================================================"
echo "Start time: $(date)"
echo "GPUs: $NUM_GPUS"
echo "Data: $DATA_DIR"
echo "============================================================"

cd $PROJECT_DIR

# =============================================================================
# PHASE 1: Train MRI Model
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE 1: Training MRI Model (~28 hours)"
echo "============================================================"

if [ -f "$CHECKPOINT_DIR/mri_best.pt" ]; then
    echo "⏭️ MRI model already trained, skipping..."
else
    echo "🚀 Starting MRI training..."

    torchrun --nproc_per_node=$NUM_GPUS \
        cloud/train_multigpu.py \
        --modality mri \
        --data-dir $DATA_DIR \
        --checkpoint-dir $CHECKPOINT_DIR \
        --epochs 50 \
        --batch-size 8 \
        --lr 1e-4 \
        2>&1 | tee $LOG_DIR/train_mri.log

    echo "✅ MRI training complete"
fi

# =============================================================================
# PHASE 2: Train CT Model
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE 2: Training CT Model (~20 hours)"
echo "============================================================"

if [ -f "$CHECKPOINT_DIR/ct_best.pt" ]; then
    echo "⏭️ CT model already trained, skipping..."
else
    echo "🚀 Starting CT training..."

    torchrun --nproc_per_node=$NUM_GPUS \
        cloud/train_multigpu.py \
        --modality ct \
        --data-dir $DATA_DIR \
        --checkpoint-dir $CHECKPOINT_DIR \
        --epochs 50 \
        --batch-size 16 \
        --lr 1e-4 \
        2>&1 | tee $LOG_DIR/train_ct.log

    echo "✅ CT training complete"
fi

# =============================================================================
# PHASE 3: Train X-ray Model
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE 3: Training X-ray Model (~18 hours)"
echo "============================================================"

if [ -f "$CHECKPOINT_DIR/xray_best.pt" ]; then
    echo "⏭️ X-ray model already trained, skipping..."
else
    echo "🚀 Starting X-ray training..."

    torchrun --nproc_per_node=$NUM_GPUS \
        cloud/train_multigpu.py \
        --modality xray \
        --data-dir $DATA_DIR \
        --checkpoint-dir $CHECKPOINT_DIR \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-4 \
        2>&1 | tee $LOG_DIR/train_xray.log

    echo "✅ X-ray training complete"
fi

# =============================================================================
# PHASE 4: Run All Experiments
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE 4: Running All Experiments (~3 hours)"
echo "============================================================"

echo "🔬 Running experiments..."
python run_all_experiments.py 2>&1 | tee $LOG_DIR/experiments.log

echo "✅ Experiments complete"

# =============================================================================
# PHASE 5: Package Results
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE 5: Packaging Results for Download"
echo "============================================================"

python cloud/package_results.py --output-dir $RESULTS_DIR

echo "✅ Results packaged"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo "✅ TRAINING PIPELINE COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results location: $RESULTS_DIR"
echo ""
echo "To download results to your local machine:"
echo "  scp -r ubuntu@<instance-ip>:$RESULTS_DIR/mri_guardian_results.tar.gz ."
echo ""
echo "Or use the download command shown below:"
ls -lh $RESULTS_DIR/*.tar.gz 2>/dev/null || echo "  (results archive will be created)"
echo ""
echo "============================================================"
