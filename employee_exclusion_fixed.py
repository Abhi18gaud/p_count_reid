import cv2
import numpy as np
import pickle
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Face recognition temporarily disabled to fix UI freezing
FACE_RECOGNITION_AVAILABLE = False
FACE_RECOGNITION_BACKEND = None
print("⚠️ Face recognition disabled - using body features only for employee exclusion")

class EmployeeExclusionSystem:
    """System to detect and exclude employees from tracking"""
    
    def __init__(self):
        """Initialize employee exclusion system"""
        self.employees_file = "employees_database.pkl"
        self.current_shifts_file = "current_employee_shifts.pkl"
        
        # Employee database: employee_id -> {face_encoding, name, role}
        self.employees_db: Dict[int, Dict] = {}
        
        # Face recognition settings
        self.face_confidence_threshold = 0.6
        
        # Body feature similarity threshold for employee verification
        self.body_similarity_threshold = 0.7
        
        # Shift duration (12 hours)
        self.shift_duration_hours = 12
        
        # Performance optimization: cache face detection results
        self.face_detection_cache = {}
        self.cache_timeout = 0.5  # Cache for 0.5 second (more aggressive)
        self.face_detection_interval = 3  # Only check every 3 frames
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
    
    def register_employee(self, face_image: np.ndarray, employee_name: str, employee_role: str) -> bool:
        """Register a new employee with face and body features"""
        try:
            employee_id = len(self.employees_db) + 1
            
            # Extract body features (color histogram, texture, etc.)
            body_features = self._extract_body_features(face_image)
            
            # Store employee data
            self.employees_db[employee_id] = {
                'name': employee_name,
                'role': employee_role,
                'registered_at': time.time(),
                'body_features': body_features
            }
            
            self._save_data()
            print(f"✅ Registered employee: {employee_name} (ID: {employee_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error registering employee: {e}")
            return False
    
    def _extract_body_features(self, image: np.ndarray) -> Dict:
        """Extract body features for employee verification"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Color histogram (HSV)
            hist_hsv = cv2.calcHist([hsv], [0, 1, 2], [50, 60, 60], cv2.HIST_CALC_RANGE)
            hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
            
            # Color histogram (LAB)
            hist_lab = cv2.calcHist([lab], [0, 1, 2], [50, 60, 60], cv2.HIST_CALC_RANGE)
            hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
            
            # LBP texture features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp = cv2.calcHist([gray], [0], [256], [0, 256], cv2.HIST_CALC_RANGE)
            lbp = cv2.normalize(lbp, lbp).flatten()
            
            # Gradient features
            grad_x = cv2.Sobel(gray, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, 0, 1, ksize=3)
            grad_hist = cv2.calcHist([grad_x, grad_y], [0, 1], [256], [0, 256], cv2.HIST_CALC_RANGE)
            grad_hist = cv2.normalize(grad_hist, grad_hist).flatten()
            
            # Color moments
            moments = cv2.moments(gray)
            color_moments = [
                moments['m00'], moments['m01'], moments['m02'], moments['m03'],
                moments['m10'], moments['m11'], moments['m12'],
                moments['mu20'], moments['mu11'], moments['mu02']
            ]
            
            return {
                'hsv_hist': hist_hsv,
                'lab_hist': hist_lab,
                'lbp_hist': lbp,
                'gradient_hist': grad_hist,
                'color_moments': color_moments,
                'mean_color': np.mean(image, axis=(0, 1)),
                'std_color': np.std(image, axis=(0, 1))
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting body features: {e}")
            return {}
    
    def detect_employee_face(self, person_crop: np.ndarray) -> Optional[int]:
        """Detect if person is a registered employee using face recognition"""
        if not self.employees_db:
            return None
            
        try:
            # Performance optimization: skip frames to reduce CPU load
            self.frame_counter += 1
            if self.frame_counter % self.face_detection_interval != 0:
                return None  # Skip face detection on this frame
            
            # Performance optimization: check cache first
            crop_hash = hash(person_crop.tobytes())
            current_time = time.time()
            
            if crop_hash in self.face_detection_cache:
                cached_result, cached_time = self.face_detection_cache[crop_hash]
                if current_time - cached_time < self.cache_timeout:
                    return cached_result
            
            # Face recognition disabled - use body features only
            result = None
            
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
            print(f"⚠️ Error in face detection: {e}")
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
    
    def is_employee(self, person_crop: np.ndarray) -> bool:
        """Check if person is a registered employee"""
        employee_id = self.detect_employee_face(person_crop)
        return employee_id is not None
    
    def get_employee_info(self, employee_id: int) -> Optional[Dict]:
        """Get employee information by ID"""
        return self.employees_db.get(employee_id)
