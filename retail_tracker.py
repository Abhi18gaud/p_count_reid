"""
Enhanced Retail Tracking System with Deep Re-ID
Modular architecture using YOLOv8m + ByteTrack + OSNet for stable person tracking
"""

import cv2
import numpy as np
import time
import math
import pickle
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import signal
import sys
import threading

STOP_SIGNAL = False

# Import professional Re-ID system
try:
    from professional_reid import ProfessionalReIDSystem, GlobalPersonGallery
    PROFESSIONAL_REID_AVAILABLE = True
    print("🧠 Professional Re-ID system available")
except ImportError:
    PROFESSIONAL_REID_AVAILABLE = False
    print("⚠️ Professional Re-ID not available, using HSV fallback")

class GlobalPersonRegistry:
    """A thread-safe global registry for person encodings and logical IDs with camera isolation."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    super(GlobalPersonRegistry, cls).__new__(cls)
                    cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            with self._lock:
                if not hasattr(self, 'initialized'):
                    self.saved_encodings = {}  # logical_id -> encoding
                    self.camera_mappings = {}   # logical_id -> {camera_id: track_id}
                    self.next_logical_id = 1
                    self.encoding_file = "global_person_encodings.pkl"
                    self.similarity_threshold = 0.70  # Adjusted threshold for better cross-camera matching
                    self.min_track_age = 10  # Minimum frames before assigning ID
                    self.initialized = True
                    self.load_encodings()

    def load_encodings(self):
        """Load saved encodings from the global file with size validation."""
        if os.path.exists(self.encoding_file):
            with open(self.encoding_file, "rb") as f:
                save_data = pickle.load(f)
                if isinstance(save_data, dict) and 'encodings' in save_data:
                    self.saved_encodings = save_data.get('encodings', {})
                    self.camera_mappings = save_data.get('camera_mappings', {})
                    self.next_logical_id = save_data.get('next_id', 1)
                    
                    # Validate encoding sizes and remove incompatible ones
                    expected_size = 512  # 8x8x8 = 512 for 3D HSV histograms
                    incompatible_ids = []
                    
                    for lid, encoding in self.saved_encodings.items():
                        if encoding is None or encoding.size != expected_size:
                            incompatible_ids.append(lid)
                    
                    if incompatible_ids:
                        print(f"🧹 Removing {len(incompatible_ids)} incompatible encodings (old format)")
                        for lid in incompatible_ids:
                            del self.saved_encodings[lid]
                            if lid in self.camera_mappings:
                                del self.camera_mappings[lid]
                        
                        # Save cleaned encodings
                        self.save_encodings()
                    
                else:
                    # Backward compatibility - old format might have different sizes
                    self.saved_encodings = {}
                    self.camera_mappings = {}
                    self.next_logical_id = 1
                    print("🔄 Old encoding format detected - starting fresh")
            print(f"🌍 Global Registry: Loaded {len(self.saved_encodings)} person encodings.")

    def save_encodings(self):
        """Save current encodings to the global file."""
        with self._lock:
            save_data = {
                'encodings': self.saved_encodings,
                'camera_mappings': self.camera_mappings,
                'next_id': self.next_logical_id
            }
            with open(self.encoding_file, "wb") as f:
                pickle.dump(save_data, f)
            print(f"🌍 Global Registry: Saved {len(self.saved_encodings)} person encodings.")

    def get_or_assign_id(self, encoding, camera_id, track_id, frame_count=0):
        """Get an existing logical ID for an encoding or assign a new one with enhanced cross-camera validation"""
        with self._lock:
            # Only assign IDs after minimum track age to avoid false matches
            if frame_count < self.min_track_age:
                return None
                
            best_lid = None
            best_score = -1

            # Check for matches with enhanced cross-camera validation
            for lid, saved_encoding in self.saved_encodings.items():
                sim = self.hsv_similarity(encoding, saved_encoding)
                
                # Get camera mapping for this person
                camera_mapping = self.camera_mappings.get(lid, {})
                person_cameras = set(camera_mapping.keys())
                
                # Adjust similarity threshold based on cross-camera matching
                adjusted_threshold = self.similarity_threshold
                
                # If person was seen in this camera before, be more strict
                if camera_id in person_cameras:
                    adjusted_threshold = self.similarity_threshold + 0.05  # More strict for same camera
                
                # If this is cross-camera matching, be slightly more lenient but still strict
                elif len(person_cameras) > 0:
                    adjusted_threshold = self.similarity_threshold - 0.03  # Slightly more lenient for cross-camera
                
                print(f"🔍 HSV Person {lid}: similarity = {sim:.3f} (threshold: {adjusted_threshold:.3f})")
                
                # Higher threshold for more reliable matching
                if sim > adjusted_threshold and sim > best_score:
                    # Additional temporal consistency check
                    if self._validate_temporal_consistency(lid, camera_id, track_id):
                        best_score = sim
                        best_lid = lid

            if best_lid is not None:
                # Assign existing ID to new camera
                if best_lid not in self.camera_mappings:
                    self.camera_mappings[best_lid] = {}
                self.camera_mappings[best_lid][camera_id] = track_id
                
                # Cross-camera success message
                cameras_seen = list(self.camera_mappings[best_lid].keys())
                print(f"🌍 HSV CROSS-CAMERA MATCH: Track {track_id} -> Person {best_lid} in {camera_id} (similarity: {best_score:.3f})")
                print(f"   Person now seen in cameras: {', '.join(cameras_seen)}")
                return best_lid
            else:
                # Create new person
                logical_id = self.next_logical_id
                self.saved_encodings[logical_id] = encoding
                self.camera_mappings[logical_id] = {camera_id: track_id}
                self.next_logical_id += 1
                print(f"🌍 HSV NEW: Person {logical_id} -> Track {track_id} in {camera_id}")
                return logical_id
    
    def _validate_temporal_consistency(self, logical_id, camera_id, track_id):
        """Validate temporal consistency to prevent rapid ID reassignments"""
        # For now, always return True - can be enhanced with tracking history
        return True

    def get_encoding(self, logical_id):
        """Get the encoding for a given logical ID."""
        with self._lock:
            return self.saved_encodings.get(logical_id)
    
    def hsv_similarity(self, e1, e2):
        """Calculate histogram correlation with multiple metrics - IMPROVED METHOD"""
        if e1 is None or e2 is None:
            return -1
        
        # Check if histograms have the same size
        if e1.size != e2.size:
            print(f"⚠️ Histogram size mismatch: {e1.size} vs {e2.size}")
            # Return low similarity for mismatched sizes
            return -0.5
        
        try:
            # Ensure both arrays are float32 for OpenCV compatibility
            e1_f32 = e1.astype(np.float32) if e1.dtype != np.float32 else e1
            e2_f32 = e2.astype(np.float32) if e2.dtype != np.float32 else e2
            
            # Use correlation as primary metric
            correlation = cv2.compareHist(e1_f32, e2_f32, cv2.HISTCMP_CORREL)
            
            # Also calculate intersection for verification
            intersection = cv2.compareHist(e1_f32, e2_f32, cv2.HISTCMP_INTERSECT)
            
            # Combine metrics (70% correlation, 30% intersection)
            combined_score = 0.7 * correlation + 0.3 * (intersection / 1000.0)
            
            return combined_score
        except cv2.error as e:
            print(f"⚠️ Histogram comparison error: {e}")
            return -0.5
    
    def get_camera_mapping(self, logical_id):
        """Get camera mapping for a logical ID."""
        with self._lock:
            return self.camera_mappings.get(logical_id, {})
    
    def remove_camera_mapping(self, logical_id, camera_id):
        """Remove camera mapping when track is lost."""
        with self._lock:
            if logical_id in self.camera_mappings:
                self.camera_mappings[logical_id].pop(camera_id, None)
                if not self.camera_mappings[logical_id]:
                    # Keep the person but clear camera mappings
                    self.camera_mappings[logical_id] = {}

def handle_sigint(sig, frame):
    global STOP_SIGNAL
    STOP_SIGNAL = True
    print("\n🛑 SIGINT received → forcing shutdown...")

signal.signal(signal.SIGINT, handle_sigint)

# Import core modules
from core.detector import PersonDetector
from core.tracker import ByteTrackManager
# from core.reid import OSNetReID  # Replaced with HSV approach
# from core.id_manager import GlobalIdentityManager  # Replaced with HSV approach

# Import existing modules
from database_manager import RetailDatabaseManager
from store_config import store_config


@dataclass
class CustomerState:
    """Customer tracking state with enhanced stability"""
    logical_id: int
    customer_db_id: Optional[int] = None
    session_id: Optional[int] = None
    current_department: Optional[str] = None
    current_visit_id: Optional[int] = None
    last_position: Optional[Tuple[int, int]] = None
    last_seen: Optional[float] = None
    entry_time: Optional[float] = None
    has_entered_store: bool = False
    departments_visited: List[str] = None
    wait_start_time: Optional[float] = None
    wait_id: Optional[int] = None
    is_being_served: bool = False
    
    def __post_init__(self):
        if self.departments_visited is None:
            self.departments_visited = []


class EnhancedRetailTracker:
    """
    Enhanced retail tracking system with deep re-identification
    Implements modular architecture for robust person tracking
    """
    
    def __init__(self, video_path: str, output_path: str, show_display: bool = True, camera_id: str = "default", stop_event=None, global_registry=None, employee_exclusion=None):
        """Initialize enhanced retail tracker with Professional Re-ID"""
        # Video setup
        self.video_path = video_path
        self.output_path = output_path
        self.show_display = show_display
        self.camera_id = camera_id  # Add camera identifier for multi-camera support
        self.stop_event = stop_event
        
        # Employee exclusion system
        self.employee_exclusion = employee_exclusion
        
        # Initialize professional Re-ID system or fallback
        if PROFESSIONAL_REID_AVAILABLE:
            print("🧠 Using Shared Professional Re-ID System...")
            # Use the shared global system passed from track-2.py
            if isinstance(global_registry, GlobalPersonGallery):
                self.global_gallery = global_registry
                self.reid_system = global_registry.reid_system
                self.use_professional_reid = True
            else:
                print("⚠️ Global system is not GlobalPersonGallery, falling back to HSV")
                self.global_registry = GlobalPersonRegistry()
                self.use_professional_reid = False
        else:
            print("⚠️ Using HSV fallback Re-ID system...")
            self.global_registry = global_registry or GlobalPersonRegistry()
            self.use_professional_reid = False
        
        # Initialize core modules
        print("🔧 Initializing core modules...")
        self.detector = PersonDetector(model_path="yolov8m.pt", device="auto")
        self.tracker = ByteTrackManager(config_path="bytetrack_retail.yaml")
        
        # Re-ID System parameters
        if self.use_professional_reid:
            self.similarity_threshold = 0.85  # More reasonable threshold for cross-camera matching
            self.embedding_dim = 512  # OSNet embedding dimension
            self.memory_timeout = 120.0  # 2 minutes (same as HSV)
            self.encoding_thresh = 0.75  # For compatibility (not used in professional Re-ID)
        else:
            self.encoding_thresh = 0.70  # Slightly lower threshold for HSV cross-camera matching
            self.memory_timeout = 120.0  # 2 minutes
        
        self.max_human_speed = 300  # pixels/second
        self.speed_mult = 1.5
        self.track_frame_counts = {}  # track_id -> frame_count
        
        # ID Management
        self.tracker_to_logical = {}  # ByteTrack ID → Logical ID
        self.logical_memory = {}      # Logical ID → {pos, time, encoding/crop}
        
        # Database connection with graceful fallback
        self.db_manager = RetailDatabaseManager()
        self.db_available = self.db_manager.connect()
        if not self.db_available:
            print("⚠️ Database unavailable - running in offline mode")
        
        # Tracking parameters
        self.save_interval = 300
        
        # Customer state management
        self.customers: Dict[int, CustomerState] = {}  # logical_id -> CustomerState
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}  # logical_id -> position history
        
        # Memory management
        self.max_customers = 1000  # Maximum customers to track
        self.max_track_history = 50  # Maximum track history length
        
        # Load saved state
        self.load_saved_state()
        
        # Video capture setup
        self.setup_video()
        
        # Statistics
        self.stats = {
            'total_footfall': 0,
            'active_sessions': 0,
            'department_visits': {},
            'service_interactions': {'attended': 0, 'unattended': 0},
            'detection_fps': 0.0,
            'reid_fps': 0.0,
            'tracking_fps': 0.0
        }
        
        if self.use_professional_reid:
            print("🚀 Enhanced Retail Tracker initialized with Professional Re-ID")
        else:
            print("🚀 Enhanced Retail Tracker initialized with HSV Re-ID")
    
    
    def extract_hsv_encoding(self, frame, bbox):
        """Extract HSV histogram encoding - IMPROVED METHOD"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Add padding to avoid edge effects
        padding = 5
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(frame.shape[1], x2 + padding)
        y2_p = min(frame.shape[0], y2 + padding)
        roi_padded = frame[y1_p:y2_p, x1_p:x2_p]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi_padded, cv2.COLOR_BGR2HSV)
        
        # Use more bins for better discrimination
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        
        return hist.flatten()
    
    def hsv_similarity(self, e1, e2):
        """Calculate histogram correlation with multiple metrics - IMPROVED METHOD"""
        if e1 is None or e2 is None:
            return -1
        
        # Check if histograms have the same size
        if e1.size != e2.size:
            print(f"⚠️ Histogram size mismatch: {e1.size} vs {e2.size}")
            # Return low similarity for mismatched sizes
            return -0.5
        
        try:
            # Ensure both arrays are float32 for OpenCV compatibility
            e1_f32 = e1.astype(np.float32) if e1.dtype != np.float32 else e1
            e2_f32 = e2.astype(np.float32) if e2.dtype != np.float32 else e2
            
            # Use correlation as primary metric
            correlation = cv2.compareHist(e1_f32, e2_f32, cv2.HISTCMP_CORREL)
            
            # Also calculate intersection for verification
            intersection = cv2.compareHist(e1_f32, e2_f32, cv2.HISTCMP_INTERSECT)
            
            # Combine metrics (70% correlation, 30% intersection)
            combined_score = 0.7 * correlation + 0.3 * (intersection / 1000.0)
            
            return combined_score
        except cv2.error as e:
            print(f"⚠️ Histogram comparison error: {e}")
            return -0.5
    
    def distance(self, a, b):
        """Calculate distance between two points"""
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def assign_logical_id_professional(self, track_id, bbox, cx, cy, frame, timestamp, frame_count=0):
        """Assign or recover logical ID using Professional Re-ID system"""
        if track_id in self.tracker_to_logical:
            return self.tracker_to_logical[track_id]
        
        # Extract person crop
        person_crop = self.extract_person_crop(frame, bbox)
        if person_crop is None:
            return None
        
        print(f"🧠 Professional Re-ID matching for Track {track_id} in {self.camera_id}...")
        
        # Query Global Gallery for a logical ID
        logical_id = self.global_gallery.match_or_create_person(
            person_crop, self.camera_id, track_id, frame_count
        )
        
        if logical_id is not None:
            print(f"🌍 PROFESSIONAL RE-ID: Track {track_id} -> Person {logical_id} in {self.camera_id}")
            self.tracker_to_logical[track_id] = logical_id
            
            # Store person crop in memory for debugging
            self.logical_memory[logical_id] = {
                "pos": (cx, cy),
                "time": timestamp,
                "crop": person_crop
            }
            
            return logical_id
        
        return None
    
    def assign_logical_id_hsv(self, track_id, bbox, cx, cy, frame, timestamp, frame_count=0):
        """Assign or recover logical ID using HSV encoding system"""
        if track_id in self.tracker_to_logical:
            return self.tracker_to_logical[track_id]
        
        # Extract HSV encoding
        encoding = self.extract_hsv_encoding(frame, bbox)
        if encoding is None:
            return None
        
        print(f"🎨 HSV Re-ID matching for Track {track_id} in {self.camera_id}...")
        
        # Query the Global Registry for a logical ID
        logical_id = self.global_registry.get_or_assign_id(
            encoding, self.camera_id, track_id, frame_count
        )
        
        if logical_id is not None:
            print(f"🌍 HSV RE-ID: Track {track_id} -> Person {logical_id} in {self.camera_id}")
            self.tracker_to_logical[track_id] = logical_id
            
            # Store encoding in memory
            self.logical_memory[logical_id] = {
                "pos": (cx, cy),
                "time": timestamp,
                "encoding": encoding
            }
            
            return logical_id
        
        return None
    
    def extract_person_crop(self, frame, bbox):
        """Extract person crop with padding"""
        x1, y1, x2, y2 = bbox
        
        # Add padding (10% of bbox size)
        padding_x = int((x2 - x1) * 0.1)
        padding_y = int((y2 - y1) * 0.1)
        
        x1_p = max(0, x1 - padding_x)
        y1_p = max(0, y1 - padding_y)
        x2_p = min(frame.shape[1], x2 + padding_x)
        y2_p = min(frame.shape[0], y2 + padding_y)
        
        crop = frame[y1_p:y2_p, x1_p:x2_p]
        
        if crop.size == 0:
            return None
        
        return crop
    
    # def cleanup_lost_tracks(self, active_tracks):
    #     """Clean up track_to_logical mapping for lost tracks"""
    #     active_track_ids = set(active_tracks.keys())
    #     lost_track_ids = []
        
    #     for track_id, logical_id in list(self.tracker_to_logical.items()):
    #         if track_id not in active_track_ids:
    #             # Track is lost, remove mapping
    #             lost_track_ids.append(track_id)
                
    #             # Remove from appropriate system
    #             if self.use_professional_reid:
    #                 self.global_gallery.remove_camera_mapping(logical_id, self.camera_id)
    #             else:
    #                 self.global_registry.remove_camera_mapping(logical_id, self.camera_id)
                
    #     for track_id in lost_track_ids:
    #         del self.tracker_to_logical[track_id]
    #         self.track_frame_counts.pop(track_id, None)
            
    #     if lost_track_ids:
    #         print(f"🗑️ Cleaned up {len(lost_track_ids)} lost track mappings in {self.camera_id}")
    
    def cleanup_expired_memory(self, timestamp):
        """Clean up expired memory (older than 2 minutes) with timer reset on re-detection"""
        expired_ids = []
        for lid, data in self.logical_memory.items():
            time_since_last_seen = timestamp - data["time"]
            if time_since_last_seen > self.memory_timeout:
                # Check if this person is currently being tracked (re-detected)
                is_currently_tracked = False
                for track_id, logical_id in self.tracker_to_logical.items():
                    if logical_id == lid:
                        # Person is still being tracked, reset their timer instead of deleting
                        self.logical_memory[lid]["time"] = timestamp
                        print(f"🔄 Person {lid} timer reset after {time_since_last_seen:.0f}s - re-detected!")
                        is_currently_tracked = True
                        break
                
                # Only mark for deletion if not currently tracked
                if not is_currently_tracked:
                    expired_ids.append(lid)
        
        for lid in expired_ids:
            del self.logical_memory[lid]
            # Clean up customer state if exists
            if lid in self.customers:
                # End session if active
                customer = self.customers[lid]
                if customer.session_id and customer.session_id != -1 and self.db_available:
                    self.db_manager.end_customer_session(customer.session_id)
                    self.stats['active_sessions'] = max(0, self.stats['active_sessions'] - 1)
                del self.customers[lid]
            
            # Clean up track history
            if lid in self.track_history:
                del self.track_history[lid]
            
            # Clean up frame counts
            track_ids_to_remove = [tid for tid, l_id in self.tracker_to_logical.items() if l_id == lid]
            for tid in track_ids_to_remove:
                del self.tracker_to_logical[tid]
                self.track_frame_counts.pop(tid, None)
            
            print(f"🗑️ Person {lid} removed after {self.memory_timeout}s timeout (not re-detected)")
    
    def setup_video(self):
        """Setup video capture and output"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open video: {self.video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 10
        
        # Create output directory
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        self.out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )
        
        # Check if video writer is properly initialized
        if not self.out.isOpened():
            raise Exception(f"Failed to initialize video writer for {self.output_path}")
        
        print(f"📹 Video setup: {self.width}x{self.height} @ {self.fps} FPS")
        print(f"📤 Output file: {self.output_path}")
        
        # Position display window to center of screen
        if self.show_display:
            window_name = f"Enhanced Retail Analytics - {self.camera_id}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            print(f"🖥️ Display window configured for {self.camera_id}")
            self.window_name = window_name
    
    def load_saved_state(self):
        """Load saved tracker state"""
        # Load appropriate global system state
        if self.use_professional_reid:
            if hasattr(self, 'global_gallery'):
                stats = self.global_gallery.get_statistics()
                print(f"🌍 Global Gallery is managing {stats['total_persons']} persons.")
            else:
                print("🌍 Global Gallery not yet initialized")
        else:
            if hasattr(self, 'global_registry'):
                print(f"🌍 Global Person Registry is managing {len(self.global_registry.saved_encodings)} encodings.")
            else:
                print("🌍 Global Registry not yet initialized")
        
        # Load global tracker state
        tracker_state_file = "global_tracker_state.pkl"
        if self.tracker.load_state(tracker_state_file):
            print(f"📂 Loaded global tracker state from {tracker_state_file}")
    
    def save_state(self):
        """Save current state to files"""
        # Save appropriate global system
        if self.use_professional_reid and hasattr(self, 'global_gallery'):
            self.global_gallery.save_gallery()
        elif hasattr(self, 'global_registry'):
            self.global_registry.save_encodings()
        
        # Save global tracker state
        tracker_state_file = "global_tracker_state.pkl"
        self.tracker.save_state(tracker_state_file)
        print(f"💾 Saved global tracker state to {tracker_state_file}")
        
        # Only print state saved message every 3rd save (every 900 frames ~ 30 seconds)
        if hasattr(self, '_save_counter'):
            self._save_counter += 1
        else:
            self._save_counter = 1
        
        if self._save_counter % 3 == 0:
            print("💾 Global state saved to files")
    
    def get_or_create_customer(self, logical_id: int, position: Tuple[int, int], timestamp: float) -> CustomerState:
        """Get existing customer or create new one with memory management"""
        if logical_id in self.customers:
            customer = self.customers[logical_id]
            customer.last_position = position
            customer.last_seen = timestamp
            return customer
        
        # Check memory limit
        if len(self.customers) >= self.max_customers:
            self.cleanup_oldest_customer()
        
        # Create new customer
        customer = CustomerState(logical_id=logical_id)
        customer.last_position = position
        customer.last_seen = timestamp
        customer.entry_time = timestamp
        
        # Get/create database record (use dummy encoding for compatibility)
        dummy_encoding = np.zeros(512)  # OSNet embedding dimension
        customer.customer_db_id = self.db_manager.get_or_create_customer(logical_id, dummy_encoding)
        
        # If database is unavailable, use logical_id as fallback
        if customer.customer_db_id == -1:
            customer.customer_db_id = logical_id
        
        self.customers[logical_id] = customer
        return customer
    
    def cleanup_oldest_customer(self):
        """Remove the oldest customer to free memory"""
        if not self.customers:
            return
        
        oldest_id = min(self.customers.keys(), key=lambda x: self.customers[x].last_seen or float('inf'))
        del self.customers[oldest_id]
        if oldest_id in self.track_history:
            del self.track_history[oldest_id]
        print(f"🗑️ Removed oldest customer {oldest_id} to free memory")
    
    def handle_entry_detection(self, customer: CustomerState, position: Tuple[int, int], timestamp: float):
        """Handle customer entry detection"""
        if customer.has_entered_store:
            return
        
        entry_point = store_config.get_entry_point_at_position(position[0], position[1])
        if entry_point:
            customer.has_entered_store = True
            
            # Create database session only if available
            if self.db_available:
                entry_point_id = 1  # Default, you should map this properly
                customer.session_id = self.db_manager.start_customer_session(
                    customer.customer_db_id, entry_point_id
                )
            else:
                customer.session_id = -1  # Fallback session ID
            
            self.stats['total_footfall'] += 1
            self.stats['active_sessions'] += 1
            
            print(f"🚪 Customer {customer.logical_id} entered via {entry_point['name']}")
    
    def handle_department_transition(self, customer: CustomerState, new_department: Optional[str], timestamp: float):
        """Handle customer moving between departments"""
        if new_department == customer.current_department:
            return
        
        # Exit current department
        if customer.current_department and customer.current_visit_id and self.db_available:
            self.db_manager.end_department_visit(customer.current_visit_id)
            customer.current_visit_id = None
            print(f"📤 Customer {customer.logical_id} left {customer.current_department}")
        
        # Enter new department
        if new_department and customer.session_id and self.db_available:
            customer.current_department = new_department
            customer.current_visit_id = self.db_manager.start_department_visit(
                customer.customer_db_id, customer.session_id, 
                self.get_department_db_id(new_department)
            )
            
            if new_department not in customer.departments_visited:
                customer.departments_visited.append(new_department)
            
            # Update statistics
            dept_stats = self.stats['department_visits']
            dept_stats[new_department] = dept_stats.get(new_department, 0) + 1
            
            print(f"📥 Customer {customer.logical_id} entered {new_department}")
    
    def handle_service_detection(self, customer: CustomerState, position: Tuple[int, int], timestamp: float):
        """Handle service interaction detection"""
        if not customer.current_department or not self.db_available:
            return
        
        dept_info = store_config.departments.get(customer.current_department)
        if not dept_info or not dept_info.get('has_service'):
            return
        
        is_near_desk = store_config.is_near_service_desk(
            position[0], position[1], customer.current_department
        )
        
        if is_near_desk and not customer.wait_start_time:
            # Start waiting
            customer.wait_start_time = timestamp
            customer.wait_id = self.db_manager.start_wait_time(
                customer.customer_db_id, customer.session_id,
                self.get_department_db_id(customer.current_department)
            )
            print(f"⏳ Customer {customer.logical_id} started waiting in {customer.current_department}")
        
        elif not is_near_desk and customer.wait_start_time:
            # End waiting (attended or unattended)
            wait_duration = timestamp - customer.wait_start_time
            is_attended = wait_duration > store_config.service_detection['wait_time_threshold']
            
            if customer.wait_id:
                self.db_manager.end_wait_time(customer.wait_id, is_served=is_attended)
            
            # Record service interaction
            self.db_manager.record_service_interaction(
                customer.customer_db_id, customer.session_id,
                self.get_department_db_id(customer.current_department),
                is_attended=is_attended
            )
            
            # Update statistics
            if is_attended:
                self.stats['service_interactions']['attended'] += 1
            else:
                self.stats['service_interactions']['unattended'] += 1
            
            customer.wait_start_time = None
            customer.wait_id = None
            
            status = "attended" if is_attended else "unattended"
            print(f"👥 Customer {customer.logical_id} {status} in {customer.current_department}")
    
    def get_department_db_id(self, department_id: str) -> int:
        """Map department ID to database ID"""
        dept_mapping = {
            'electronics': 1,
            'clothing': 2,
            'groceries': 3,
            'furniture': 4,
            'checkout': 5
        }
        return dept_mapping.get(department_id, 1)
    
    def update_customer_state(self, customer: CustomerState, timestamp: float):
        """Update customer state and handle timeouts"""
        # Check for session timeout
        if (customer.last_seen and 
            timestamp - customer.last_seen > store_config.tracking_params['session_timeout']):
            
            if customer.session_id and customer.session_id != -1 and self.db_available:
                self.db_manager.end_customer_session(customer.session_id)
                self.stats['active_sessions'] -= 1
            
            # Remove customer
            del self.customers[customer.logical_id]
            if customer.logical_id in self.track_history:
                del self.track_history[customer.logical_id]
            print(f"👋 Customer {customer.logical_id} session ended")
    
    def draw_zones(self, frame: np.ndarray):
        """Draw department and entry point zones"""
        # Draw departments
        for dept_id, dept in store_config.departments.items():
            color = dept['color']
            cv2.rectangle(frame, (dept['x1'], dept['y1']), (dept['x2'], dept['y2']), color, 2)
            
            # Draw department name
            cv2.putText(frame, dept['name'], 
                       (dept['x1'] + 5, dept['y1'] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw service desk if exists
            if dept.get('has_service') and 'service_desk' in dept:
                desk = dept['service_desk']
                cv2.circle(frame, (desk['x'], desk['y']), 8, color, -1)
                cv2.circle(frame, (desk['x'], desk['y']), 15, color, 2)
        
        # Draw entry points
        for ep_id, ep in store_config.entry_points.items():
            if ep['is_entrance']:
                color = ep['color']
                cv2.rectangle(frame, (ep['x1'], ep['y1']), (ep['x2'], ep['y2']), color, 2)
                cv2.putText(frame, ep['name'],
                           (ep['x1'] + 5, ep['y1'] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_customer_info_professional(self, frame, logical_id, bbox, time_in_memory):
        """Draw customer information with professional Re-ID data"""
        x1, y1, x2, y2 = bbox
        
        # Check if this person is an employee and get their name
        person_crop = frame[y1:y2, x1:x2]
        is_employee = False
        employee_name = None
        
        if self.employee_exclusion and person_crop.size > 0:
            is_employee = self.employee_exclusion.is_employee(person_crop)
            if is_employee:
                employee_name = self.employee_exclusion.get_employee_name(is_employee)
        
        # Display employee name or customer ID
        if is_employee and employee_name:
            status_text = f"EMP: {employee_name}"
            color = (0, 0, 255)  # Red for employees
        else:
            status_text = f"ID {logical_id}"
            color = (0, 255, 0)  # Green for customers
        
        if hasattr(self, 'customers') and logical_id in self.customers:
            customer = self.customers[logical_id]
            if customer.current_department:
                status_text += f" [{customer.current_department}]"
            if customer.wait_start_time:
                status_text += " "
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, status_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_customer_info_hsv(self, frame, customer, bbox, time_in_memory):
        """Draw customer info with HSV time indicators"""
        x1, y1, x2, y2 = bbox
        
        # Color based on time in memory
        if time_in_memory < 5:
            color = (0, 255, 0)  # Green - recent
        elif time_in_memory < 30:
            color = (0, 255, 255)  # Yellow - medium
        else:
            color = (0, 0, 255)  # Red - old
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw customer info
        info_text = f"ID {customer.logical_id} ({time_in_memory:.0f}s)"
        if customer.current_department:
            info_text += f" | {customer.current_department}"
        if customer.wait_start_time:
            info_text += " ⏳"
        
        cv2.putText(frame, info_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw trajectory
        if customer.logical_id in self.track_history:
            trajectory = self.track_history[customer.logical_id]
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i-1], trajectory[i], color, 2)
    
    def draw_statistics(self, frame: np.ndarray):
        """Draw tracking statistics"""
        y_offset = 30
        stats_text = [
            f"Footfall: {self.stats['total_footfall']}",
            f"Active: {self.stats['active_sessions']}",
            f"Attended: {self.stats['service_interactions']['attended']}",
            f"Unattended: {self.stats['service_interactions']['unattended']}",
            f"FPS: {self.stats['detection_fps']:.1f}"
        ]
        
        for text in stats_text:
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 25
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process a single frame with HSV Re-ID tracking"""
        timestamp = time.time()
        start_time = timestamp
        
        # Step 1: Detection with tracking
        yaml_path = os.path.join(os.path.dirname(__file__), "bytetrack_retail.yaml")
        detection_results = self.detector.detect_with_tracking(frame, yaml_path)
        detection_time = time.time() - start_time
        self.stats['detection_fps'] = 1.0 / max(detection_time, 0.001) if detection_time > 0 else 0.0
        
        # Step 2: Update ByteTrack
        if detection_results:
            active_tracks = self.tracker.update_tracks(detection_results, timestamp)
        else:
            active_tracks = {}
        
        # Step 3: Employee Exclusion Check and ID Assignment
        employee_tracks = {}  # Store employee tracks for display
        
        print(f"🔍 DEBUG: Processing {len(active_tracks)} active tracks")
        
        for track_id, track in list(active_tracks.items()):
            if track.lost_count > 0:
                continue  # Skip lost tracks
            
            print(f"🔍 DEBUG: Processing track {track_id}")
            
            # Extract person crop for employee detection
            x1, y1, x2, y2 = map(int, track.bbox)
            person_crop = frame[y1:y2, x1:x2]
            
            print(f"🔍 DEBUG: Person crop size: {person_crop.shape if person_crop.size > 0 else 'Empty'}")
            
            # Check if this person is an employee BEFORE ID assignment
            is_employee = False
            employee_name = None
            if self.employee_exclusion and person_crop.size > 0:
                print(f"🔍 DEBUG: Calling employee exclusion for track {track_id}")
                is_employee = self.employee_exclusion.is_employee(person_crop)
                if is_employee:
                    employee_name = self.employee_exclusion.get_employee_name(is_employee)
                    print(f"🔍 DEBUG: Employee {employee_name} detected, is_employee={is_employee}")
            
            # If employee detected, store for display but remove from customer processing
            if is_employee:
                if employee_name:
                    print(f"👤 Employee detected: {employee_name} (track {track_id}), excluding from customer tracking but showing on display")
                else:
                    print(f"👤 Employee detected (track {track_id}), excluding from customer tracking but showing on display")
                
                # Mark as employee for display purposes
                track.is_employee = True
                track.employee_name = employee_name
                employee_tracks[track_id] = track  # Store for display
                
                # Remove from active tracks to prevent customer counting and ID assignment
                del active_tracks[track_id]
                continue
            
            # Update frame count for this track
            self.track_frame_counts[track_id] = self.track_frame_counts.get(track_id, 0) + 1
            frame_count = self.track_frame_counts[track_id]
            
            # Assign/recover logical ID using appropriate Re-ID system
            if self.use_professional_reid:
                logical_id = self.assign_logical_id_professional(
                    track_id, track.bbox, track.center[0], track.center[1], frame, timestamp, frame_count
                )
            else:
                logical_id = self.assign_logical_id_hsv(
                    track_id, track.bbox, track.center[0], track.center[1], frame, timestamp, frame_count
                )
            
            if logical_id is None:
                continue  # Skip if no ID assigned yet
            
            # Update memory with appropriate data
            if self.use_professional_reid:
                # For professional Re-ID, we already stored crop in assign_logical_id_professional
                pass
            else:
                # For HSV, update encoding
                encoding = self.extract_hsv_encoding(frame, track.bbox)
                if encoding is not None:
                    self.logical_memory[logical_id] = {
                        "pos": track.center,
                        "time": timestamp,
                        "encoding": encoding
                    }
            
            # Get or create customer
            customer = self.get_or_create_customer(logical_id, track.center, timestamp)
            
            # Update trajectory with memory limit
            self.track_history.setdefault(logical_id, []).append(track.center)
            if len(self.track_history[logical_id]) > self.max_track_history:
                self.track_history[logical_id].pop(0)
            
            # Handle retail events
            self.handle_entry_detection(customer, track.center, timestamp)
            
            # Check department
            current_dept = store_config.get_department_at_position(track.center[0], track.center[1])
            dept_name = current_dept['id'] if current_dept else None
            self.handle_department_transition(customer, dept_name, timestamp)
            
            # Handle service detection
            self.handle_service_detection(customer, track.center, timestamp)
            
            # Draw customer info with HSV indicators
            time_in_memory = timestamp - self.logical_memory[logical_id]["time"]
            self.draw_customer_info_hsv(frame, customer, track.bbox, time_in_memory)
        
        # Step 4: Display all tracks (including employees)
        # First, display employee tracks with names
        for track_id, track in employee_tracks.items():
            if track.lost_count > 0:
                continue  # Skip lost tracks
            
            # Draw employee with name
            x1, y1, x2, y2 = map(int, track.bbox)
            employee_name = getattr(track, 'employee_name', 'Employee')
            
            # Draw employee box and name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
            cv2.putText(frame, f"EMP: {employee_name}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Then, display customer tracks
        for track_id, track in active_tracks.items():
            if track.lost_count > 0:
                continue  # Skip lost tracks
            
            # This is a customer track that got processed above
            logical_id = self.tracker_to_logical.get(track_id)
            if logical_id is not None:
                time_in_memory = timestamp - self.logical_memory[logical_id]["time"]
                self.draw_customer_info_professional(frame, logical_id, track.bbox, time_in_memory)
        
        # Step 5: Clean up expired memory and lost tracks
        self.cleanup_expired_memory(timestamp)
        # self.cleanup_lost_tracks(active_tracks)
        
        # Periodically save global system and cleanup
        if frame_idx % self.save_interval == 0:
            if self.use_professional_reid and hasattr(self, 'global_gallery'):
                self.global_gallery.save_gallery()
            elif hasattr(self, 'global_registry'):
                self.global_registry.save_encodings()
            # Additional cleanup every 5 minutes
            if frame_idx % (self.save_interval * 10) == 0:
                self.cleanup_old_data()
        
        # Draw zones and statistics
        self.draw_zones(frame)
        self.draw_statistics(frame)
        
        return frame
    
    def run(self):
        """Main tracking loop"""
        frame_idx = 0
        frames_written = 0
        start_time = time.time()
        
        print("🎬 Starting enhanced video processing with deep re-identification...")
        
        # while self.cap.isOpened():
        while self.cap.isOpened():
            # Check for shared stop signal
            if self.stop_event and self.stop_event.is_set():
                print(f"🛑 Stop signal received for {self.camera_id}. Shutting down...")
                break
            if STOP_SIGNAL:
                break
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_idx += 1
            frame_start = time.time()
            
            # Process frame
            processed_frame = self.process_frame(frame, frame_idx)
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            fps = 1 / frame_time if frame_time > 0 else 0.0
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                       (self.width - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Write output with validation
            if self.out.isOpened():
                self.out.write(processed_frame)
                frames_written += 1
            else:
                print(f"⚠️ Video writer not open for {self.camera_id} - frame not written")
            
            # Display
            if self.show_display:
                cv2.imshow(self.window_name, processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if self.stop_event and self.stop_event.is_set():
                    print(f"🛑 Stop signal received for {self.camera_id}. Shutting down...")
                    break
                if STOP_SIGNAL:
                    break
                if key == 27:  # ESC key
                    if self.stop_event:
                        self.stop_event.set()
                    break
                elif key == ord('q'):  # Q key
                    print(f"🛑 'q' pressed in {self.camera_id}. Signaling shutdown...")
                    if self.stop_event:
                        self.stop_event.set()
                    break
                elif key == ord(' '):  # Space key to pause
                    print("⏸️ Paused - press space to resume")
                    cv2.waitKey(0)
            
            # Periodically save state
            if frame_idx % self.save_interval == 0:
                self.save_state()
        
        # Cleanup - ensure all outputs are saved
        print(f"🔄 Cleaning up {self.camera_id}...")
        self.save_state()  # Save state before releasing resources
        
        # Save global system one final time
        if self.use_professional_reid and hasattr(self, 'global_gallery'):
            self.global_gallery.save_gallery()
            print(f"💾 Global gallery saved by {self.camera_id}")
        elif hasattr(self, 'global_registry'):
            self.global_registry.save_encodings()
            print(f"💾 Global registry saved by {self.camera_id}")
        
        # Force video writer to flush and release properly
        if hasattr(self, 'out') and self.out.isOpened():
            print(f"💾 Flushing video output for {self.camera_id}...")
            self.out.release()  # This forces the video file to be written
            
            # Verify the output file was created and has content
            import os
            if os.path.exists(self.output_path):
                file_size = os.path.getsize(self.output_path)
                print(f"✅ Output file saved: {self.output_path} ({file_size} bytes)")
                if file_size < 1024:  # Less than 1KB is probably empty
                    print(f"⚠️ Warning: Output file seems very small ({file_size} bytes)")
            else:
                print(f"❌ Error: Output file not found at {self.output_path}")
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
        cv2.destroyAllWindows()
        
        # Disconnect from database gracefully
        if self.db_available:
            self.db_manager.disconnect()
        
        print(f"✅ {self.camera_id} cleanup completed")
        
        # Print final statistics
        total_time = time.time() - start_time
        print(f"\n📊 Enhanced Processing Complete!")
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print(f"📹 Frames processed: {frame_idx}")
        print(f"� Frames written: {frames_written}")
        print(f"�� Total footfall: {self.stats['total_footfall']}")
        print(f"🏪 Department visits: {dict(self.stats['department_visits'])}")
        print(f"👥 Service interactions: {self.stats['service_interactions']}")
        
    def cleanup_old_data(self):
        """Clean up old data to prevent memory leaks"""
        # Clean up old track frame counts
        old_track_ids = [tid for tid, count in self.track_frame_counts.items() if count > 10000]
        for tid in old_track_ids:
            self.track_frame_counts.pop(tid, None)
        
        # Limit track history length
        for lid in self.track_history:
            if len(self.track_history[lid]) > 100:
                self.track_history[lid] = self.track_history[lid][-50:]
        
        print(f"🧹 Memory cleanup completed")
        
        # Print module statistics
        print(f"\n🔧 Module Statistics:")
        print(f"   Detector: {self.detector.get_performance_stats()}")
        print(f"   Tracker: {self.tracker.get_statistics()}")
        
        if self.use_professional_reid and hasattr(self, 'global_gallery'):
            stats = self.global_gallery.get_statistics()
            print(f"   Global Gallery: {stats['total_persons']} persons tracked")
        elif hasattr(self, 'global_registry'):
            print(f"   Global Registry: {len(self.global_registry.saved_encodings)} persons tracked")
        
        print(f"   Memory Usage: {len(self.customers)} active customers")


if __name__ == "__main__":
    # Initialize and run enhanced tracker
    tracker = EnhancedRetailTracker(
        video_path="0",
        output_path="outputs/retail_analytics_enhanced.mp4",
        show_display=True,
        camera_id="default"
    )
    tracker.run()
