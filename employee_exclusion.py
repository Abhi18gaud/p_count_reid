import cv2
import numpy as np
import pickle
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Try to import DeepFace as primary face recognition solution
try:
    # Configure GPU environment variables
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    
    from deepface import DeepFace
    FACE_RECOGNITION_AVAILABLE = False
    FACE_RECOGNITION_BACKEND = "DeepFace"
    
    # Check if TensorFlow has GPU support
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Using DeepFace with TensorFlow GPU: {len(gpus)} device(s)")
    else:
        print("⚠️ TensorFlow CPU-only detected - DeepFace will use CPU (causing lag)")
        print("💡 To fix lag: Install tensorflow-gpu or use PyTorch-based face recognition")
    
except ImportError:
    # Fallback to face_recognition
    try:
        import face_recognition
        FACE_RECOGNITION_AVAILABLE = False
        FACE_RECOGNITION_BACKEND = "face_recognition"
        print("⚠️ DeepFace not available, using face_recognition fallback")
    except ImportError:
        FACE_RECOGNITION_AVAILABLE = False
        FACE_RECOGNITION_BACKEND = None
        print("❌ No face recognition library available")

class EmployeeExclusionSystem:
    """System to detect and exclude employees from tracking"""
    
    def __init__(self):
        """Initialize employee exclusion system"""
        self.employees_file = "employees_database.pkl"
        self.current_shifts_file = "current_employee_shifts.pkl"
        
        # Employee database: employee_id -> {face_encoding, name, role}
        self.employees_db: Dict[int, Dict] = {}
        
        # Face recognition settings
        self.face_confidence_threshold = 0.8
        
        # Body feature similarity threshold for employee verification
        self.body_similarity_threshold = 0.8
        
        # Shift duration (12 hours)
        self.shift_duration_hours = 12
        
        # Performance optimization: cache face detection results
        self.face_detection_cache = {}
        self.cache_timeout = 0.5  # Cache for 0.5 second (more aggressive)
        self.face_detection_interval = 1  # Check every frame for debugging
        self.frame_counter = 0
        
        print("👥 Employee Exclusion System initialized")
        self._load_data()
        
    def _load_data(self):
        """Load employee database and current shifts"""
        try:
            # Load employee database
            if os.path.exists(self.employees_file):
                with open(self.employees_file, "rb") as f:
                    self.employees_db = pickle.load(f)
                    print(f"👥 Loaded {len(self.employees_db)} registered employees")
            
            # Load current shifts
            if os.path.exists(self.current_shifts_file):
                with open(self.current_shifts_file, "rb") as f:
                    self.current_shifts = pickle.load(f)
                    print(f"⏰ Loaded {len(self.current_shifts)} active employee shifts")
            else:
                self.current_shifts = {}
                    
        except Exception as e:
            print(f"⚠️ Error loading employee data: {e}")
            self.employees_db = {}
            self.current_shifts = {}
    
    def _save_data(self):
        """Save employee database and current shifts"""
        try:
            # Save employee database
            with open(self.employees_file, "wb") as f:
                pickle.dump(self.employees_db, f)
            
            # Save current shifts
            with open(self.current_shifts_file, "wb") as f:
                pickle.dump(self.current_shifts, f)
                print("💾 Saved employee data")
        except Exception as e:
            print(f"⚠️ Error saving employee data: {e}")
    
    def _cleanup_expired_shifts(self):
        """Remove shifts older than 12 hours"""
        current_time = time.time()
        cutoff_time = current_time - (self.shift_duration_hours * 3600)
        
        expired_shifts = []
        for emp_id, shift_data in list(self.current_shifts.items()):
            if shift_data['start_time'] < cutoff_time:
                expired_shifts.append(emp_id)
        
        # Remove expired shifts
        for emp_id in expired_shifts:
            del self.current_shifts[emp_id]
            print(f"🕐 Expired shift for employee {emp_id}")
    
    def register_employee(self, face_image: np.ndarray, employee_name: str, employee_role: str = "Employee") -> Optional[int]:
        """Register a new employee using OSNet for face recognition"""
        try:
            return self._register_employee_osnet(face_image, employee_name, employee_role)
        except Exception as e:
            print(f"⚠️ Error registering employee: {e}")
            return None
    
    def _register_employee_osnet(self, face_image: np.ndarray, employee_name: str, employee_role: str) -> Optional[int]:
        """Register employee using OSNet for face embedding"""
        try:
            # Import OSNet system
            from professional_reid import ProfessionalReIDSystem
            
            # Initialize OSNet system if not already done
            if not hasattr(self, 'osnet_system'):
                print("🧠 Initializing OSNet for employee registration...")
                self.osnet_system = ProfessionalReIDSystem("osnet_x1_0_msmt17.pth")
            
            print(f"🧠 OSNET REGISTRATION: Processing face image of shape {face_image.shape}")
            
            # Get OSNet embedding for face image
            embedding = self.osnet_system.extract_embedding(face_image)
            if embedding is None:
                print("🧠 OSNET REGISTRATION: Failed to extract embedding")
                return None
            
            print(f"🧠 OSNET REGISTRATION: Extracted embedding of shape {embedding.shape}")
            
            # Generate new employee ID
            employee_id = max(self.employees_db.keys(), default=0) + 1
            
            # Store employee data with OSNet embedding
            self.employees_db[employee_id] = {
                'name': employee_name,
                'role': employee_role,
                'face_encoding': embedding,
                'registration_time': time.time(),
                'detection_method': 'OSNet'
            }
            
            # Save to file
            self._save_data()
            
            print(f"✅ Employee registered with OSNet: {employee_name} (ID: {employee_id})")
            return employee_id
            
        except Exception as e:
            print(f"⚠️ OSNet registration error: {e}")
            return None
    
    def _register_employee_deepface(self, face_image: np.ndarray, employee_name: str, employee_role: str) -> Optional[np.ndarray]:
        """Register employee face encoding using DeepFace"""
        try:
            # Convert BGR to RGB for DeepFace
            if face_image.shape[2] == 3:
                rgb_image = face_image[:, :, ::-1]
            else:
                rgb_image = face_image
            
            # Extract face embedding using DeepFace
            embedding_objs = DeepFace.represent(
                rgb_image,
                enforce_detection=False,
                detector_backend='retinaface',
                model_name='VGG-Face',
                align=True,
                normalization='Facenet2018'
            )
            
            if not embedding_objs:
                print("❌ No face detected in the provided image")
                return None
            
            # Get the first face embedding
            face_embedding = np.array(embedding_objs[0]["embedding"])
            
            # Store model info for consistency
            print(f"🔍 Registered embedding shape: {face_embedding.shape}")
            
            return face_embedding
            
        except Exception as e:
            print(f"⚠️ DeepFace registration error: {e}")
            return None
    
    def _extract_body_features(self, image: np.ndarray) -> Dict:
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Color histogram (HSV) - Fixed syntax for newer OpenCV
            hist_h = cv2.calcHist(images=[hsv], channels=[0], mask=None, histSize=[50], ranges=[0, 180])
            hist_s = cv2.calcHist(images=[hsv], channels=[1], mask=None, histSize=[60], ranges=[0, 256])
            hist_v = cv2.calcHist(images=[hsv], channels=[2], mask=None, histSize=[60], ranges=[0, 256])
            hist_hsv = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
            
            # Color histogram (LAB) - Fixed syntax for newer OpenCV
            hist_l = cv2.calcHist(images=[lab], channels=[0], mask=None, histSize=[50], ranges=[0, 256])
            hist_a = cv2.calcHist(images=[lab], channels=[1], mask=None, histSize=[60], ranges=[0, 256])
            hist_b = cv2.calcHist(images=[lab], channels=[2], mask=None, histSize=[60], ranges=[0, 256])
            hist_lab = np.concatenate([hist_l.flatten(), hist_a.flatten(), hist_b.flatten()])
            hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
            
            # LBP texture features - Fixed syntax for newer OpenCV
            lbp = cv2.calcHist(images=[gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            lbp = cv2.normalize(lbp, lbp).flatten()
            
            # Gradient features - Fixed syntax and data type for newer OpenCV
            grad_x = cv2.Sobel(src=gray, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=3)
            grad_y = cv2.Sobel(src=gray, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=3)
            
            # Convert gradients to proper format for histogram
            grad_x = np.uint8(np.absolute(grad_x))
            grad_y = np.uint8(np.absolute(grad_y))
            
            grad_hist_x = cv2.calcHist(images=[grad_x], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            grad_hist_y = cv2.calcHist(images=[grad_y], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            grad_hist = np.concatenate([grad_hist_x.flatten(), grad_hist_y.flatten()])
            grad_hist = cv2.normalize(grad_hist, grad_hist).flatten()
            
            # Color moments
            moments = cv2.moments(gray)
            color_moments = [
                moments['m00'], moments['m01'], moments['m02'], moments['m03'],
                moments['m10'], moments['m11'], moments['m12'],
                moments['mu20'], moments['mu11'], moments['mu02']
            ]
            
            # Store all features
            features = {
                'color_histogram_hsv': hist_hsv,
                'color_histogram_lab': hist_lab,
                'texture_lbp': lbp,
                'gradient_hist': grad_hist,
                'color_moments': color_moments,
                'mean_color': np.mean(image, axis=(0, 1)),
                'std_color': np.std(image, axis=(0, 1))
            }
            
            return features
            
        except Exception as e:
            print(f"⚠️ Error extracting body features: {e}")
            return {}
    
            return None
            
        except Exception as e:
            print(f"⚠️ DeepFace detection error: {e}")
            return None

    def detect_employee_face(self, person_crop: np.ndarray) -> Optional[int]:
        """Detect employee using OSNet (osnet_x1_0_msmt17.pth) for face detection"""
        if not self.employees_db:
            return None
        
        print(f"🔍 EMPLOYEE DEBUG: Checking person crop of shape {person_crop.shape}")
        
        try:
            # Performance optimization: skip frames to reduce CPU load
            self.frame_counter += 1
            if self.frame_counter % self.face_detection_interval != 0:
                print(f"🔍 EMPLOYEE DEBUG: Skipping frame {self.frame_counter} (interval: {self.face_detection_interval})")
                return None  # Skip face detection on this frame
            
            # Performance optimization: check cache first
            crop_hash = hash(person_crop.tobytes())
            current_time = time.time()
            
            if crop_hash in self.face_detection_cache:
                cached_result, cached_time = self.face_detection_cache[crop_hash]
                if current_time - cached_time < self.cache_timeout:
                    return cached_result
            
            # Use OSNet for face detection instead of DeepFace
            result = self._detect_employee_osnet(person_crop)
            
            # Cache result
            self.face_detection_cache[crop_hash] = (result, current_time)
            
            # Clean old cache entries
            if len(self.face_detection_cache) > 50:
                self.face_detection_cache = {
                    k: v for k, v in self.face_detection_cache.items() 
                    if current_time - v[1] < self.cache_timeout * 2
                }
            
            return result
                
        except Exception as e:
            print(f"⚠️ Error in employee detection: {e}")
            return None
    
    def _detect_employee_osnet(self, person_crop: np.ndarray) -> Optional[int]:
        """Detect employee using OSNet model for face recognition"""
        try:
            # Import OSNet system
            from professional_reid import ProfessionalReIDSystem
            
            # Initialize OSNet system if not already done
            if not hasattr(self, 'osnet_system'):
                print("🧠 Initializing OSNet for employee face detection...")
                self.osnet_system = ProfessionalReIDSystem("osnet_x1_0_msmt17.pth")
            
            # Extract face region from person crop (simple approach - use upper portion)
            height, width = person_crop.shape[:2]
            face_region = person_crop[:int(height * 0.4), :]  # Top 40% for face
            
            print(f"🧠 OSNET DEBUG: Processing face region of shape {face_region.shape}")
            
            # Get OSNet embedding for face region
            embedding = self.osnet_system.extract_embedding(face_region)
            if embedding is None:
                print("🧠 OSNET DEBUG: Failed to extract embedding")
                return None
            
            print(f"🧠 OSNET DEBUG: Extracted embedding of shape {embedding.shape}")
            
            # Compare with registered employee face embeddings
            best_match_id = None
            best_similarity = 0.0
            
            for emp_id, emp_data in self.employees_db.items():
                if 'face_encoding' not in emp_data:
                    continue
                
                known_embedding = emp_data['face_encoding']
                
                # Calculate cosine similarity
                similarity = np.dot(embedding, known_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
                )
                
                print(f"🧠 OSNET DEBUG: Employee {emp_id} similarity: {similarity:.3f}")
                
                if similarity > self.face_confidence_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = emp_id
            
            if best_match_id:
                print(f"👤 Employee OSNet detected: {self.employees_db[best_match_id]['name']} (similarity: {best_similarity:.3f})")
                return best_match_id
            else:
                print("🧠 OSNET DEBUG: No matching employee found")
                return None
                
        except Exception as e:
            print(f"⚠️ OSNet detection error: {e}")
            return None
    
    def update_employee_shift(self, employee_id: int, person_crop: np.ndarray):
        """Update employee's current shift with new body features"""
        if employee_id not in self.employees_db:
            return False
            
        try:
            # Extract new body features
            new_features = self._extract_body_features(person_crop)
            
            # Get current shift data
            if employee_id not in self.current_shifts:
                self.current_shifts[employee_id] = {
                    'start_time': time.time(),
                    'body_features': new_features,
                    'last_update': time.time()
                }
            else:
                # Update existing shift
                self.current_shifts[employee_id]['body_features'] = new_features
                self.current_shifts[employee_id]['last_update'] = time.time()
            
            self._save_data()
            print(f"🔄 Updated shift for employee {employee_id} with new body features")
            return True
            
        except Exception as e:
            print(f"⚠️ Error updating employee shift: {e}")
            return False
    
    def is_employee(self, person_crop: np.ndarray) -> Optional[int]:
        """Check if person is a registered employee using face recognition first, then body features"""
        if not self.employees_db:
            return None
        
        print(f"🔍 EMPLOYEE DEBUG: Checking person crop of shape {person_crop.shape}")
        
        try:
            # First try face recognition (only for initial detection)
            employee_id = self.detect_employee_face(person_crop)
            if employee_id:
                print(f"🔍 EMPLOYEE DEBUG: Face recognition found employee {employee_id}")
                # When face is detected, capture and save body features for 12 hours
                self.update_employee_shift(employee_id, person_crop)
                print(f"📸 EMPLOYEE DEBUG: Captured and saved body features for employee {employee_id} for 12 hours")
                return employee_id
            
            # Fallback to body features matching (using OSNet-like approach)
            print(f"🔍 EMPLOYEE DEBUG: Face recognition failed, trying body features matching")
            employee_id = self.detect_employee_body(person_crop)
            if employee_id:
                print(f"🔍 EMPLOYEE DEBUG: Body features matched employee {employee_id}")
                return employee_id
            
            print(f"🔍 EMPLOYEE DEBUG: No employee found")
            return None
                
        except Exception as e:
            print(f"⚠️ Error in employee detection: {e}")
            return None
    
    def detect_employee_body(self, person_crop: np.ndarray) -> Optional[int]:
        """Detect employee using body features matching (OSNet-like approach)"""
        if not self.current_shifts:
            print("🔍 BODY DEBUG: No active employee shifts found")
            return None
        
        print(f"🔍 BODY DEBUG: Checking body features against {len(self.current_shifts)} active shifts")
        
        try:
            # Extract current body features
            current_features = self._extract_body_features(person_crop)
            if not current_features:
                print("🔍 BODY DEBUG: Failed to extract body features")
                return None
            
            # Compare with stored body features
            best_match_id = None
            best_similarity = 0.0
            
            for emp_id, shift_data in self.current_shifts.items():
                stored_features = shift_data.get('body_features', {})
                if not stored_features:
                    continue
                
                # Calculate similarity for each feature type
                similarities = []
                
                # Color histogram similarity
                if 'color_histogram_hsv' in current_features and 'color_histogram_hsv' in stored_features:
                    current_hist = current_features['color_histogram_hsv']
                    stored_hist = stored_features['color_histogram_hsv']
                    
                    # Calculate histogram similarity
                    hist_similarity = cv2.compareHist(
                        np.array(current_hist, dtype=np.float32), 
                        np.array(stored_hist, dtype=np.float32), 
                        cv2.HISTCMP_CORREL
                    )
                    similarities.append(hist_similarity)
                    print(f"🔍 BODY DEBUG: Employee {emp_id} histogram similarity: {hist_similarity:.3f}")
                
                # Texture similarity
                if 'texture_lbp' in current_features and 'texture_lbp' in stored_features:
                    current_lbp = current_features['texture_lbp']
                    stored_lbp = stored_features['texture_lbp']
                    
                    # Calculate LBP histogram similarity
                    lbp_similarity = cv2.compareHist(
                        np.array(current_lbp, dtype=np.float32), 
                        np.array(stored_lbp, dtype=np.float32), 
                        cv2.HISTCMP_CORREL
                    )
                    similarities.append(lbp_similarity)
                    print(f"🔍 BODY DEBUG: Employee {emp_id} texture similarity: {lbp_similarity:.3f}")
                
                # Gradient similarity
                if 'gradient_hist' in current_features and 'gradient_hist' in stored_features:
                    current_grad = current_features['gradient_hist']
                    stored_grad = stored_features['gradient_hist']
                    
                    # Calculate gradient similarity
                    grad_similarity = cv2.compareHist(
                        np.array(current_grad, dtype=np.float32), 
                        np.array(stored_grad, dtype=np.float32), 
                        cv2.HISTCMP_CORREL
                    )
                    similarities.append(grad_similarity)
                    print(f"🔍 BODY DEBUG: Employee {emp_id} gradient similarity: {grad_similarity:.3f}")
                
                # Calculate overall similarity
                if similarities:
                    overall_similarity = np.mean(similarities)
                    print(f"🔍 BODY DEBUG: Employee {emp_id} overall similarity: {overall_similarity:.3f}")
                    
                    if overall_similarity > best_similarity and overall_similarity > self.body_similarity_threshold:
                        best_similarity = overall_similarity
                        best_match_id = emp_id
            
            if best_match_id:
                print(f"🎯 BODY DEBUG: Best match found - Employee {best_match_id} (similarity: {best_similarity:.3f})")
                return best_match_id
            else:
                print("🔍 BODY DEBUG: No matching employee found")
                return None
                
        except Exception as e:
            print(f"⚠️ Error in body feature detection: {e}")
            return None
    
    def remove_employee(self, employee_id: int) -> bool:
        """Remove an employee from the database"""
        try:
            if employee_id not in self.employees_db:
                print(f"❌ Employee {employee_id} not found")
                return False
            
            employee_name = self.employees_db[employee_id]['name']
            
            # Remove from employees database
            del self.employees_db[employee_id]
            
            # Remove from active shifts if present
            if employee_id in self.current_shifts:
                del self.current_shifts[employee_id]
                print(f"🕐 Removed active shift for employee {employee_name}")
            
            # Save updated data
            self._save_data()
            
            print(f"✅ Successfully removed employee {employee_name} (ID: {employee_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error removing employee: {e}")
            return False
    
    def get_employee_name(self, employee_id: int) -> Optional[str]:
        """Get employee name by ID"""
        if employee_id in self.employees_db:
            return self.employees_db[employee_id]['name']
        return None
    
    def get_employee_id(self, person_crop: np.ndarray) -> Optional[int]:
        """Get employee ID if detected"""
        return self.detect_employee_face(person_crop)
