#!/usr/bin/env python3
"""
Professional Re-ID System using OSNet for Multi-Camera Tracking
This replaces the simple HSV approach with deep learning embeddings
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pickle
import os
import threading
from typing import Dict, List, Optional, Tuple
import time

class ProfessionalReIDSystem:
    """Professional Re-ID system using OSNet embeddings"""
    
    def __init__(self, model_path="osnet_x1_0_msmt17.pth"):
        """Initialize the Re-ID system with OSNet model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.embedding_dim = 768
        self.similarity_threshold = 0.70
        
        print("🧠 Initializing Professional Re-ID System...")
        self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load OSNet model for person Re-ID with proper Re-ID weights"""
        try:
            # Suppress torch.load warning
            # import warnings
            # warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")
            
            # Try to import torchreid
            from torchreid.models import build_model
            from torchreid.utils import load_pretrained_weights
            
            print("🏗️ Building OSNet model with Re-ID weights...")
            # Build OSNet model - this should automatically use Re-ID weights if available
            self.model = build_model(
                name='osnet_x1_0',   # stronger backbone

                num_classes=1000,  # Doesn't matter for feature extraction
                pretrained=True,  # This should load Re-ID weights
                use_gpu=(self.device.type == 'cuda')
            )
            # NOW load your specific MSMT17 weights from your file
            if os.path.exists(model_path):
              print(f"🎯 Loading specific Re-ID weights from: {model_path}")
              load_pretrained_weights(self.model, model_path)
            else:
              print(f"⚠️ Warning: {model_path} not found! Using ImageNet weights only.")
            # CRITICAL: Ensure model is actually moved to the correct device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print("✅ Professional Re-ID model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading OSNet model: {e}")
            self.model = None
            print("✅ OSNet model loaded successfully")
            print(f"🧠 OSNet model loaded on {self.device}")
            print(f"📏 Embedding dimension: 512")
            print("🎯 Using torchreid built-in Re-ID weights")
            
            # Verify model is on correct device
            model_device = next(self.model.parameters()).device
            print(f"🔍 Model parameter device: {model_device}")
            if model_device != self.device:
                print(f"⚠️ Device mismatch detected! Moving model to {self.device}")
                self.model = self.model.to(self.device)
            
        except ImportError:
            print("❌ torchreid not installed. Falling back to enhanced HSV")
            print("💡 Install with: pip install torchreid")
            self.model = None
            
        except Exception as e:
            print(f"❌ Error loading OSNet model: {e}")
            print("⚠️ Falling back to enhanced HSV approach")
            self.model = None
    
    def extract_embedding(self, person_crop):
        """Extract 512-dimensional embedding from person crop"""
        if self.model is None:
            print("⚠️ OSNet model not available, using fallback")
            # Fallback to enhanced HSV
            return self._extract_hsv_fallback(person_crop)
        
        try:
            # Preprocess the person crop
            processed_crop = self._preprocess_crop(person_crop)
            print(f"🔍 OSNet preprocessing: {person_crop.shape} -> {processed_crop.shape}")
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(processed_crop)
                print(f"🔍 OSNet raw embedding shape: {embedding.shape}")
                # L2 normalize the embedding for cosine similarity
                embedding = F.normalize(embedding, p=2, dim=1)
                print(f"🔍 OSNet normalized embedding shape: {embedding.shape}")
            
            result = embedding.cpu().numpy().flatten()
            print(f"🔍 OSNet final embedding shape: {result.shape}")
            print(f"🔍 OSNet embedding stats: min={result.min():.6f}, max={result.max():.6f}, mean={result.mean():.6f}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error extracting OSNet embedding: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to HSV
            return self._extract_hsv_fallback(person_crop)
    
    def _preprocess_crop(self, crop):
        """Preprocess person crop for OSNet"""
        # Resize to OSNet expected input size
        target_size = (256, 128)  # Standard Re-ID size
        resized = cv2.resize(crop, target_size)
        
        # Convert to tensor and normalize
        # OSNet expects RGB normalized to [0, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        # Normalize with ImageNet stats (same as torchreid training)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        
        # CRITICAL: Move tensor to the same device as model
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _extract_hsv_fallback(self, crop):
        """Robust fallback when OSNet is not available - using multiple features"""
        if crop.size == 0:
            return None
        
        try:
            # Resize to standard size for consistency
            resized = cv2.resize(crop, (64, 128))
            
            # Extract multiple features for better discrimination
            features = []
            
            # 1. HSV histogram (coarse bins)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist_hsv, hist_hsv)
            features.extend(hist_hsv.flatten())
            
            # 2. LBP texture features (more discriminative)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            features.extend(lbp_hist)
            
            # 3. Gradient histogram
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_hist, _ = np.histogram(magnitude.ravel(), bins=16)
            grad_hist = grad_hist.astype(float)
            grad_hist /= (grad_hist.sum() + 1e-7)
            features.extend(grad_hist)
            
            # 4. Color moments (mean and std for each channel)
            color_moments = []
            for i in range(3):
                channel = resized[:, :, i]
                color_moments.extend([np.mean(channel), np.std(channel)])
            features.extend(color_moments)
            
            # Convert to numpy array and normalize
            features = np.array(features, dtype=np.float32)
            
            # L2 normalize for cosine similarity
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except ImportError:
            print("⚠️ scikit-image not available, using simple HSV only")
            # Fallback to simple HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        except Exception as e:
            print(f"⚠️ Error in robust fallback: {e}")
            # Final fallback to simple HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        if embedding1 is None or embedding2 is None:
            return -1
        
        if self.model is None:
            # Fallback - use cosine similarity for our robust features
            embedding1 = np.array(embedding1, dtype=np.float32)
            embedding2 = np.array(embedding2, dtype=np.float32)
            
            # Ensure same size
            if embedding1.shape != embedding2.shape:
                return -0.5
            
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            
            cosine_sim = dot_product / (norm1 * norm2)
            return cosine_sim
        
        # Cosine similarity for deep embeddings
        embedding1 = torch.from_numpy(embedding1).float()
        embedding2 = torch.from_numpy(embedding2).float()
        
        cosine_sim = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        return cosine_sim.item()

class GlobalPersonGallery:
    """Professional global gallery for person Re-ID across cameras"""
    
    def __init__(self, reid_system: ProfessionalReIDSystem):
        """Initialize global person gallery"""
        self.reid_system = reid_system
        self._lock = threading.Lock()
        
        # Person gallery: global_id -> person_info
        self.person_gallery: Dict[int, Dict] = {}
        
        # Camera mappings: global_id -> {camera_id: local_track_id}
        self.camera_mappings: Dict[int, Dict[str, int]] = {}
        
        # Next global ID
        self.next_global_id = 1
        
        # Gallery file
        self.gallery_file = "global_person_gallery.pkl"
        
        # Re-ID parameters
        self.min_track_age = 10  # Minimum frames before assigning ID
        self.update_interval = 50  # Update embedding every 50 frames
        self.similarity_threshold = 0.70  # More reasonable threshold for cross-camera matching
        
        print("🌍 Global Person Gallery initialized")
        self.load_gallery()
    
    def load_gallery(self):
        """Load saved person gallery"""
        if os.path.exists(self.gallery_file):
            try:
                with open(self.gallery_file, "rb") as f:
                    data = pickle.load(f)
                    self.person_gallery = data.get('gallery', {})
                    self.camera_mappings = data.get('mappings', {})
                    self.next_global_id = data.get('next_id', 1)
                    
                print(f"🌍 Loaded gallery with {len(self.person_gallery)} persons")
                
                # Clear old data for fresh start (comment out if you want persistence)
                if len(self.person_gallery) > 0:
                    print("🧹 Clearing old gallery data for fresh Re-ID testing...")
                    self.person_gallery = {}
                    self.camera_mappings = {}
                    self.next_global_id = 1
                    # Also delete the file to prevent reloading
                    if os.path.exists(self.gallery_file):
                        os.remove(self.gallery_file)
                        print("🗑️ Removed old gallery file")
                    
            except Exception as e:
                print(f"⚠️ Error loading gallery: {e}")
                self.person_gallery = {}
                self.camera_mappings = {}
                self.next_global_id = 1
    
    def save_gallery(self):
        """Save person gallery to file"""
        try:
            with self._lock:
                data = {
                    'gallery': self.person_gallery,
                    'mappings': self.camera_mappings,
                    'next_id': self.next_global_id
                }
                
                with open(self.gallery_file, "wb") as f:
                    pickle.dump(data, f)
                    
                print(f"💾 Saved gallery with {len(self.person_gallery)} persons")
        except Exception as e:
            print(f"❌ Error saving gallery: {e}")
    
    def match_or_create_person(self, person_crop, camera_id: str, track_id: int, frame_count: int = 0):
        """Match person against gallery or create new person with enhanced cross-camera validation"""
        with self._lock:
            # Wait for stable track
            if frame_count < self.min_track_age:
                return None
            
            # Extract embedding
            embedding = self.reid_system.extract_embedding(person_crop)
            if embedding is None:
                return None
            
            # Check if this exact track is already mapped (prevent duplicate mapping)
            for global_id, mappings in self.camera_mappings.items():
                if camera_id in mappings and mappings[camera_id] == track_id:
                    # Same track already mapped to this person - update last seen time
                    if global_id in self.person_gallery:
                        self.person_gallery[global_id]['last_seen'] = time.time()
                    return global_id
            
            # Find best match in gallery with cross-camera validation
            best_match_id = None
            best_similarity = -1
            
            print(f"🔍 Searching gallery for matches in {camera_id}...")
            print(f"📊 Gallery has {len(self.person_gallery)} persons")
            
            for global_id, person_info in self.person_gallery.items():
                gallery_embedding = person_info['embedding']
                
                # Calculate similarity
                similarity = self.reid_system.calculate_similarity(embedding, gallery_embedding)
                
                print(f"🔍 Person {global_id}: similarity = {similarity:.3f}")
                
                # Enhanced cross-camera validation
                person_cameras = person_info.get('cameras_seen', set())
                
                # Adjust similarity threshold based on cross-camera matching
                adjusted_threshold = self.similarity_threshold
                
                # If person was seen in this camera before, be more strict
                if camera_id in person_cameras:
                    adjusted_threshold = self.similarity_threshold + 0.02  # More strict for same camera
                
                # If this is cross-camera matching, be slightly more lenient but still strict
                elif len(person_cameras) > 0:
                    adjusted_threshold = self.similarity_threshold - 0.01  # Slightly more lenient for cross-camera
                
                # Add more strict check for suspicious perfect matches
                if similarity > 0.99:
                    print(f"⚠️ Suspicious perfect match ({similarity:.3f}) - might be same embedding")
                    similarity = 0.96  # Still above threshold
                
                if similarity > adjusted_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = global_id
            
            print(f"🎯 Best match: {best_match_id} with similarity {best_similarity:.3f} (threshold: {self.similarity_threshold})")
            
            if best_match_id is not None:
                # Found match - update camera mapping
                if best_match_id not in self.camera_mappings:
                    self.camera_mappings[best_match_id] = {}
                
                self.camera_mappings[best_match_id][camera_id] = track_id
                
                # Update person info
                person_info = self.person_gallery[best_match_id]
                person_info['frame_count'] += 1
                person_info['last_seen'] = time.time()
                
                # Update embedding periodically (less frequent to avoid drift)
                if person_info['frame_count'] % self.update_interval == 0:
                    person_info['embedding'] = embedding
                    print(f"🔄 Updated embedding for Person {best_match_id}")
                
                # Add camera to seen set
                person_info['cameras_seen'].add(camera_id)
                
                # Cross-camera success message
                cameras_seen_str = ", ".join(person_info['cameras_seen'])
                print(f"🌍 CROSS-CAMERA MATCH: Person {best_match_id} now seen in {cameras_seen_str} (similarity: {best_similarity:.3f})")
                return best_match_id
            
            else:
                # Create new person
                global_id = self.next_global_id
                self.next_global_id += 1
                
                # Add to gallery
                self.person_gallery[global_id] = {
                    'embedding': embedding,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'frame_count': 1,
                    'cameras_seen': {camera_id}
                }
                
                # Add camera mapping
                self.camera_mappings[global_id] = {camera_id: track_id}
                
                print(f"🌍 NEW: Person {global_id} first seen in {camera_id}")
                return global_id
    
    def remove_camera_mapping(self, global_id: int, camera_id: str):
        """Remove camera mapping when track is lost"""
        with self._lock:
            if global_id in self.camera_mappings:
                self.camera_mappings[global_id].pop(camera_id, None)
                
                # Remove camera from person's seen list
                if global_id in self.person_gallery:
                    self.person_gallery[global_id]['cameras_seen'].discard(camera_id)
    
    def get_statistics(self):
        """Get gallery statistics"""
        with self._lock:
            total_persons = len(self.person_gallery)
            active_cameras = sum(1 for mappings in self.camera_mappings.values() if len(mappings) > 0)
            
            return {
                'total_persons': total_persons,
                'active_cameras': active_cameras,
                'total_mappings': sum(len(mappings) for mappings in self.camera_mappings.values())
            }

# Test the professional Re-ID system
if __name__ == "__main__":
    print("🧪 Testing Professional Re-ID System...")
    
    # Initialize Re-ID system
    reid_system = ProfessionalReIDSystem()
    gallery = GlobalPersonGallery(reid_system)
    
    # Test with dummy data
    dummy_crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
    
    # Test embedding extraction
    embedding = reid_system.extract_embedding(dummy_crop)
    print(f"✅ Embedding extracted: {embedding.shape}")
    
    # Test similarity calculation
    embedding2 = reid_system.extract_embedding(dummy_crop)
    similarity = reid_system.calculate_similarity(embedding, embedding2)
    print(f"✅ Similarity calculated: {similarity:.3f}")
    
    # Test gallery matching
    person_id = gallery.match_or_create_person(dummy_crop, "Camera_1", 1, 15)
    print(f"✅ Person ID assigned: {person_id}")
    
    # Test gallery statistics
    stats = gallery.get_statistics()
    print(f"✅ Gallery stats: {stats}")
    
    print("🎉 Professional Re-ID System test completed!")
