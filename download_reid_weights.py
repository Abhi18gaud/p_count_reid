#!/usr/bin/env python3
"""
Download OSNet Re-ID weights for proper person re-identification
"""

import os
import sys
import torch

# Add current directory and virtual environment to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to add virtual environment site-packages
venv_site_packages = os.path.join(current_dir, 'track', 'lib', 'site-packages')
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

def download_reid_weights():
    """Download OSNet weights trained on Re-ID datasets"""
    print("🔄 Downloading OSNet Re-ID weights...")
    print(f"🐍 Python path: {sys.executable}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    try:
        print("🔍 Checking torchreid availability...")
        import torchreid
        print(f"✅ torchreid found: {torchreid.__version__}")
        
        # Try different import methods
        try:
            from torchreid.utils.model_utils import load_pretrained_weights
            from torchreid.models import build_model
            print("✅ Using torchreid.utils.model_utils")
        except ImportError:
            try:
                # Alternative import path
                from torchreid.models import build_model
                print("✅ Using direct model build (will download weights)")
                load_pretrained_weights = None  # We'll use alternative method
            except ImportError:
                print("❌ Could not import torchreid models")
                return False
        
        # Create OSNet model
        model = build_model(
            name='osnet_x0_25',
            num_classes=1000,
            pretrained=True,
            use_gpu=torch.cuda.is_available()
        )
        
        # Try to download and save MSMT17 weights (best for generalization)
        try:
            if load_pretrained_weights:
                print("📥 Downloading OSNet MSMT17 weights...")
                load_pretrained_weights(model, 'osnet_x0_25_msmt17')
                
                # Save the weights
                torch.save(model.state_dict(), 'osnet_x0_25_msmt17.pth')
                print("✅ Downloaded and saved OSNet MSMT17 weights")
                return True
            else:
                # Use direct download method
                return download_weights_directly()
                
        except Exception as e:
            print(f"⚠️ Failed to download MSMT17: {e}")
            
            try:
                if load_pretrained_weights:
                    print("📥 Downloading OSNet Market1501 weights...")
                    load_pretrained_weights(model, 'osnet_x0_25_market1501')
                    
                    # Save the weights
                    torch.save(model.state_dict(), 'osnet_x0_25_market1501.pth')
                    print("✅ Downloaded and saved OSNet Market1501 weights")
                    return True
                else:
                    return download_weights_directly()
                    
            except Exception as e2:
                print(f"⚠️ Failed to download Market1501: {e2}")
                return download_weights_directly()

def download_weights_directly():
    """Download weights directly using gdown"""
    print("📥 Attempting direct download...")
    
    try:
        import gdown
        print("✅ gdown available")
        
        # MSMT17 weights URL (you may need to find the correct URL)
        urls = {
            'osnet_x0_25_msmt17.pth': 'https://drive.google.com/uc?id=1DvqOc8v4j2qL8a5_6X7i7v4n3m2k1l0j',  # Example URL
            'osnet_x0_25_market1501.pth': 'https://drive.google.com/uc?id=1DvqOc8v4j2qL8a5_6X7i7v4n3m2k1l0j'  # Example URL
        }
        
        for filename, url in urls.items():
            if not os.path.exists(filename):
                print(f"📥 Downloading {filename}...")
                try:
                    gdown.download(url, filename, quiet=False)
                    if os.path.exists(filename):
                        print(f"✅ Downloaded {filename}")
                        return True
                except Exception as e:
                    print(f"⚠️ Failed to download {filename}: {e}")
                    continue
        
        print("❌ All direct downloads failed")
        return False
        
    except ImportError:
        print("❌ gdown not available")
        print("💡 Install with: pip install gdown")
        return False
    except Exception as e:
        print(f"❌ Direct download failed: {e}")
        return False
                
    except ImportError as e:
        print(f"❌ torchreid import failed: {e}")
        print(f"🔍 Available packages in site-packages:")
        venv_site_packages = os.path.join(current_dir, 'track', 'lib', 'site-packages')
        if os.path.exists(venv_site_packages):
            try:
                packages = [p for p in os.listdir(venv_site_packages) if 'torch' in p.lower()]
                print(f"   Found: {packages}")
            except:
                print("   Could not list packages")
        print("💡 Try installing with: pip install torchreid")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
        return False

if __name__ == "__main__":
    success = download_reid_weights()
    if success:
        print("\n🎉 Re-ID weights downloaded successfully!")
        print("💡 Now run: python track-2.py")
    else:
        print("\n❌ Failed to download Re-ID weights")
        print("💡 The system will fall back to ImageNet weights (suboptimal)")
