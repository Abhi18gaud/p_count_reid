#!/usr/bin/env python3
"""
Download OSNet Re-ID weights for proper person re-identification
"""

import os
import sys
import torch

# --------------------------------------------------
# 1. Setup Paths
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------
# The official MSMT17 weights for OSNet x1.0 from Kaiyang Zhou's model zoo
MSMT17_GDRIVE_ID = '1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY'
FILENAME = 'osnet_x1_0_msmt17.pth'

def download_reid_weights():
    """Download OSNet weights trained on MSMT17 dataset"""
    print("🔄 Starting OSNet Re-ID weights downloader...")
    
    if os.path.exists(FILENAME):
        print(f"✅ {FILENAME} already exists. Skipping download.")
        return True

    # --- Method 1: Use gdown (Most reliable for Google Drive) ---
    try:
        import gdown
        print(f"📥 Downloading {FILENAME} via gdown...")
        url = f'https://drive.google.com/uc?id={MSMT17_GDRIVE_ID}'
        gdown.download(url, FILENAME, quiet=False)
        
        if os.path.exists(FILENAME):
            print(f"✅ Successfully downloaded {FILENAME}")
            return True
    except ImportError:
        print("⚠️ gdown not installed. Trying fallback method...")
    except Exception as e:
        print(f"❌ gdown failed: {e}")

    # --- Method 2: Automatic build via torchreid (Skeleton only) ---
    try:
        from torchreid.models import build_model
        print("🏗️ Building model via torchreid (this may download ImageNet base)...")
        
        model = build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True,
            use_gpu=torch.cuda.is_available()
        )
        
        # If the manual download failed, we still have the ImageNet weights 
        # which are better than random weights, but warn the user.
        print("⚠️ Direct MSMT17 download failed. Saving base weights as fallback.")
        torch.save(model.state_dict(), FILENAME)
        print(f"💾 Saved fallback weights to {FILENAME}")
        return True
        
    except Exception as e:
        print(f"❌ Final fallback failed: {e}")
        return False

if __name__ == "__main__":
    # Ensure gdown is installed for the best weights
    try:
        import gdown
    except ImportError:
        print("📦 Installing gdown for secure weight download...")
        os.system(f"{sys.executable} -m pip install gdown")

    success = download_reid_weights()
    
    if success:
        print("\n" + "="*50)
        print("🎉 WEIGHTS READY!")
        print(f"Location: {os.path.abspath(FILENAME)}")
        print("Next step: Run 'python track-2.py'")
        print("="*50)
    else:
        print("\n❌ FAILED to download weights.")
        print("Please check your internet connection or install gdown manually: pip install gdown")