#!/bin/bash
# =============================================================================
# MRI-GUARDIAN Cloud Setup Script
# =============================================================================
# Run this FIRST when you start your Lambda Labs / cloud instance
#
# Usage:
#   chmod +x setup_cloud.sh
#   ./setup_cloud.sh
#
# This script:
#   1. Installs all dependencies
#   2. Downloads FastMRI directly from AWS S3 (fast, cloud-to-cloud)
#   3. Downloads CT and X-ray datasets
#   4. Prepares everything for training
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "MRI-GUARDIAN Cloud Setup"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# Configuration
DATA_DIR="/data/medical_imaging"
PROJECT_DIR="/home/ubuntu/mri-guardian"
FASTMRI_BUCKET="s3://fastmri-dataset"

# Create directories
echo "📁 Creating directories..."
mkdir -p $DATA_DIR/{mri,ct,xray}
mkdir -p $PROJECT_DIR

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq git wget unzip awscli pigz

# Clone or copy project (assuming you've uploaded it)
echo ""
echo "📂 Setting up project..."
if [ -d "$PROJECT_DIR/mri_guardian" ]; then
    echo "   Project already exists"
else
    echo "   Please upload your project to $PROJECT_DIR"
    echo "   Use: scp -r /path/to/MRI\\ Scan/* ubuntu@<instance-ip>:$PROJECT_DIR/"
fi

# Install Python dependencies
echo ""
echo "🐍 Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install --quiet -r $PROJECT_DIR/requirements.txt
pip install --quiet awscli boto3 h5py nibabel pydicom

# =============================================================================
# DOWNLOAD FASTMRI (MRI Data)
# =============================================================================
echo ""
echo "============================================================"
echo "📥 Downloading FastMRI Brain Dataset"
echo "============================================================"

# FastMRI is on AWS S3 - cloud-to-cloud transfer is fast!
cd $DATA_DIR/mri

# Check if already downloaded
if [ -d "brain_multicoil_train" ]; then
    echo "   ⏭️ FastMRI train already exists, skipping..."
else
    echo "   Downloading brain_multicoil_train.tar.gz (~800GB)..."
    echo "   This takes ~30-60 minutes (cloud-to-cloud transfer)"
    aws s3 cp $FASTMRI_BUCKET/brain_multicoil_train.tar.gz . --no-sign-request --only-show-errors

    echo "   Extracting..."
    pigz -dc brain_multicoil_train.tar.gz | tar xf -
    rm brain_multicoil_train.tar.gz  # Save space
    echo "   ✅ FastMRI train complete"
fi

if [ -d "brain_multicoil_val" ]; then
    echo "   ⏭️ FastMRI val already exists, skipping..."
else
    echo "   Downloading brain_multicoil_val.tar.gz (~100GB)..."
    aws s3 cp $FASTMRI_BUCKET/brain_multicoil_val.tar.gz . --no-sign-request --only-show-errors

    echo "   Extracting..."
    pigz -dc brain_multicoil_val.tar.gz | tar xf -
    rm brain_multicoil_val.tar.gz
    echo "   ✅ FastMRI val complete"
fi

# =============================================================================
# DOWNLOAD CT DATA
# =============================================================================
echo ""
echo "============================================================"
echo "📥 Downloading CT Datasets"
echo "============================================================"

cd $DATA_DIR/ct

# COVID-CT (small, from GitHub)
if [ -d "COVID-CT" ]; then
    echo "   ⏭️ COVID-CT already exists, skipping..."
else
    echo "   Downloading COVID-CT..."
    git clone --depth 1 https://github.com/UCSD-AI4H/COVID-CT.git
    echo "   ✅ COVID-CT complete"
fi

# DeepLesion (larger CT dataset)
if [ -d "DeepLesion" ]; then
    echo "   ⏭️ DeepLesion already exists, skipping..."
else
    echo "   Downloading DeepLesion..."
    # DeepLesion is hosted on NIH Box - need to download manually or use kaggle
    # For now, we'll use the COVID-CT as primary and add instructions
    echo "   ⚠️ DeepLesion requires manual download from NIH"
    echo "   See: https://nihcc.app.box.com/v/DeepLesion"
    mkdir -p DeepLesion
fi

# =============================================================================
# DOWNLOAD X-RAY DATA
# =============================================================================
echo ""
echo "============================================================"
echo "📥 Downloading X-ray Datasets"
echo "============================================================"

cd $DATA_DIR/xray

# COVID Chest X-ray (small, from GitHub)
if [ -d "covid-chestxray-dataset" ]; then
    echo "   ⏭️ COVID-CXR already exists, skipping..."
else
    echo "   Downloading COVID Chest X-ray..."
    git clone --depth 1 https://github.com/ieee8023/covid-chestxray-dataset.git
    echo "   ✅ COVID-CXR complete"
fi

# ChestX-ray14 (larger)
if [ -d "ChestXray14" ]; then
    echo "   ⏭️ ChestX-ray14 already exists, skipping..."
else
    echo "   Downloading ChestX-ray14..."
    # This is on NIH Box, we'll download via direct links
    mkdir -p ChestXray14
    cd ChestXray14

    # Download images (12 zip files)
    for i in $(seq -w 1 12); do
        if [ ! -f "images_0${i}.tar.gz" ]; then
            echo "   Downloading images_0${i}.tar.gz..."
            wget -q "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz" -O images_0${i}.tar.gz 2>/dev/null || true
        fi
    done

    # Extract
    for f in *.tar.gz; do
        if [ -f "$f" ]; then
            tar -xzf $f
            rm $f
        fi
    done

    cd ..
    echo "   ✅ ChestX-ray14 complete"
fi

# =============================================================================
# VERIFY DOWNLOADS
# =============================================================================
echo ""
echo "============================================================"
echo "📊 Verifying Downloads"
echo "============================================================"

echo ""
echo "MRI Data:"
ls -lh $DATA_DIR/mri/ 2>/dev/null || echo "   No MRI data found"
MRI_COUNT=$(find $DATA_DIR/mri -name "*.h5" 2>/dev/null | wc -l)
echo "   H5 files: $MRI_COUNT"

echo ""
echo "CT Data:"
ls -lh $DATA_DIR/ct/ 2>/dev/null || echo "   No CT data found"
CT_COUNT=$(find $DATA_DIR/ct -name "*.png" -o -name "*.dcm" 2>/dev/null | wc -l)
echo "   Image files: $CT_COUNT"

echo ""
echo "X-ray Data:"
ls -lh $DATA_DIR/xray/ 2>/dev/null || echo "   No X-ray data found"
XRAY_COUNT=$(find $DATA_DIR/xray -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
echo "   Image files: $XRAY_COUNT"

# =============================================================================
# CREATE DATA MANIFEST
# =============================================================================
echo ""
echo "📋 Creating data manifest..."

cat > $DATA_DIR/manifest.json << EOF
{
    "created": "$(date -Iseconds)",
    "datasets": {
        "mri": {
            "path": "$DATA_DIR/mri",
            "train_dir": "$DATA_DIR/mri/brain_multicoil_train",
            "val_dir": "$DATA_DIR/mri/brain_multicoil_val",
            "n_files": $MRI_COUNT
        },
        "ct": {
            "path": "$DATA_DIR/ct",
            "n_files": $CT_COUNT
        },
        "xray": {
            "path": "$DATA_DIR/xray",
            "n_files": $XRAY_COUNT
        }
    }
}
EOF

echo "   Manifest saved to $DATA_DIR/manifest.json"

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Data directory: $DATA_DIR"
echo "Project directory: $PROJECT_DIR"
echo ""
echo "Disk usage:"
df -h /data 2>/dev/null || df -h /
echo ""
echo "Next steps:"
echo "  1. Upload your project code (if not done):"
echo "     scp -r 'MRI Scan/*' ubuntu@<ip>:$PROJECT_DIR/"
echo ""
echo "  2. Start training:"
echo "     cd $PROJECT_DIR"
echo "     ./cloud/train_all.sh"
echo ""
echo "End time: $(date)"
echo "============================================================"
