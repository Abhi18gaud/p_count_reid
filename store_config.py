"""
Store Configuration
Defines department zones, entry points, and store layout
"""

from typing import Dict, List, Tuple
import json

class StoreConfig:
    """Store layout and zone configuration"""
    
    def __init__(self):
        # Video dimensions (adjust based on your video)
        self.video_width = 1000
        self.video_height = 500
        
        # Entry points (store entrances)
        self.entry_points = {
            'main_entrance': {
                'name': 'Main Entrance',
                'x1': 50, 'y1': 400, 'x2': 150, 'y2': 480,
                'is_entrance': True,
                'color': (0, 255, 0),  # Green
                'description': 'Primary store entrance'
            },
            'side_entrance': {
                'name': 'Side Entrance', 
                'x1': 850, 'y1': 400, 'x2': 950, 'y2': 480,
                'is_entrance': True,
                'color': (0, 255, 0),  # Green
                'description': 'Secondary entrance'
            }
        }
        
        # Departments (store sections)
        self.departments = {
            'electronics': {
                'name': 'Electronics',
                'x1': 50, 'y1': 200, 'x2': 250, 'y2': 380,
                'color': (255, 0, 0),  # Red
                'description': 'TVs, phones, computers',
                'has_service': True,
                'service_desk': {'x': 150, 'y': 290}
            },
            'clothing': {
                'name': 'Clothing',
                'x1': 300, 'y1': 200, 'x2': 500, 'y2': 380,
                'color': (0, 0, 255),  # Blue
                'description': 'Men and women clothing',
                'has_service': True,
                'service_desk': {'x': 400, 'y': 290}
            },
            'groceries': {
                'name': 'Groceries',
                'x1': 550, 'y1': 200, 'x2': 750, 'y2': 380,
                'color': (0, 255, 255),  # Yellow
                'description': 'Food and beverages',
                'has_service': False
            },
            'furniture': {
                'name': 'Furniture',
                'x1': 800, 'y1': 200, 'x2': 950, 'y2': 380,
                'color': (255, 0, 255),  # Magenta
                'description': 'Home furniture and decor',
                'has_service': True,
                'service_desk': {'x': 875, 'y': 290}
            },
            'checkout': {
                'name': 'Checkout',
                'x1': 400, 'y1': 400, 'x2': 600, 'y2': 480,
                'color': (255, 255, 0),  # Cyan
                'description': 'Payment counters',
                'has_service': True,
                'service_desk': {'x': 500, 'y': 440}
            }
        }
        
        # Service detection parameters
        self.service_detection = {
            'proximity_threshold': 50,  # pixels
            'wait_time_threshold': 30,   # seconds before considered waiting
            'interaction_timeout': 300,  # 5 minutes max interaction time
            'employee_detection_zones': {
                'electronics': [(100, 250), (200, 330)],
                'clothing': [(350, 250), (450, 330)],
                'furniture': [(825, 250), (925, 330)],
                'checkout': [(375, 420), (625, 470)]
            }
        }
        
        # Tracking parameters
        self.tracking_params = {
            'min_dwell_time': 5,        # seconds to count as visit
            'session_timeout': 600,     # 10 minutes to end session
            'max_idle_time': 120,       # 2 minutes before considered idle
            'reentry_cooldown': 30      # seconds before same customer can re-enter
        }
    
    def get_department_at_position(self, x: int, y: int) -> Dict:
        """Get department at given position"""
        for dept_id, dept_info in self.departments.items():
            if (dept_info['x1'] <= x <= dept_info['x2'] and 
                dept_info['y1'] <= y <= dept_info['y2']):
                return {'id': dept_id, **dept_info}
        return None
    
    def get_entry_point_at_position(self, x: int, y: int) -> Dict:
        """Get entry point at given position"""
        for ep_id, ep_info in self.entry_points.items():
            if (ep_info['x1'] <= x <= ep_info['x2'] and 
                ep_info['y1'] <= y <= ep_info['y2'] and
                ep_info['is_entrance']):
                return {'id': ep_id, **ep_info}
        return None
    
    def is_near_service_desk(self, x: int, y: int, department_id: str) -> bool:
        """Check if position is near service desk"""
        dept = self.departments.get(department_id)
        if not dept or not dept.get('has_service'):
            return False
        
        desk = dept.get('service_desk')
        if not desk:
            return False
        
        distance = ((x - desk['x'])**2 + (y - desk['y'])**2)**0.5
        return distance <= self.service_detection['proximity_threshold']
    
    def get_department_color(self, department_id: str) -> Tuple[int, int, int]:
        """Get department color for visualization"""
        dept = self.departments.get(department_id)
        return dept.get('color', (128, 128, 128)) if dept else (128, 128, 128)
    
    def get_all_departments(self) -> List[Dict]:
        """Get all departments as list"""
        return [{'id': k, **v} for k, v in self.departments.items()]
    
    def get_all_entry_points(self) -> List[Dict]:
        """Get all entry points as list"""
        return [{'id': k, **v} for k, v in self.entry_points.items()]
    
    def save_to_file(self, filename: str = 'store_layout.json'):
        """Save configuration to JSON file"""
        config_data = {
            'video_width': self.video_width,
            'video_height': self.video_height,
            'entry_points': self.entry_points,
            'departments': self.departments,
            'service_detection': self.service_detection,
            'tracking_params': self.tracking_params
        }
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"✅ Store configuration saved to {filename}")
    
    def load_from_file(self, filename: str = 'store_layout.json'):
        """Load configuration from JSON file"""
        try:
            with open(filename, 'r') as f:
                config_data = json.load(f)
            
            self.video_width = config_data.get('video_width', 1000)
            self.video_height = config_data.get('video_height', 500)
            self.entry_points = config_data.get('entry_points', {})
            self.departments = config_data.get('departments', {})
            self.service_detection = config_data.get('service_detection', {})
            self.tracking_params = config_data.get('tracking_params', {})
            
            print(f"✅ Store configuration loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"❌ Configuration file {filename} not found")
            return False
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return False
    
    def validate_zones(self) -> List[str]:
        """Validate zone configurations and return issues"""
        issues = []
        
        # Check for overlapping departments
        dept_list = list(self.departments.values())
        for i, dept1 in enumerate(dept_list):
            for dept2 in dept_list[i+1:]:
                if self._zones_overlap(dept1, dept2):
                    issues.append(f"Overlapping departments: {dept1['name']} and {dept2['name']}")
        
        # Check if zones are within video bounds
        all_zones = list(self.departments.values()) + list(self.entry_points.values())
        for zone in all_zones:
            if (zone['x1'] < 0 or zone['x2'] > self.video_width or
                zone['y1'] < 0 or zone['y2'] > self.video_height):
                issues.append(f"Zone {zone['name']} extends outside video bounds")
        
        # Check service desk validity
        for dept_id, dept in self.departments.items():
            if dept.get('has_service'):
                desk = dept.get('service_desk')
                if not desk:
                    issues.append(f"Department {dept['name']} has_service=True but no service_desk")
                elif not (dept['x1'] <= desk['x'] <= dept['x2'] and 
                         dept['y1'] <= desk['y'] <= dept['y2']):
                    issues.append(f"Service desk for {dept['name']} is outside department bounds")
        
        return issues
    
    def _zones_overlap(self, zone1: Dict, zone2: Dict) -> bool:
        """Check if two zones overlap"""
        return not (zone1['x2'] <= zone2['x1'] or zone2['x2'] <= zone1['x1'] or
                   zone1['y2'] <= zone2['y1'] or zone2['y2'] <= zone1['y1'])
    
    def get_zone_boundaries(self) -> Dict:
        """Get all zone boundaries for drawing"""
        boundaries = {
            'departments': [],
            'entry_points': []
        }
        
        for dept_id, dept in self.departments.items():
            boundaries['departments'].append({
                'id': dept_id,
                'name': dept['name'],
                'coords': (dept['x1'], dept['y1'], dept['x2'], dept['y2']),
                'color': dept['color']
            })
        
        for ep_id, ep in self.entry_points.items():
            if ep['is_entrance']:
                boundaries['entry_points'].append({
                    'id': ep_id,
                    'name': ep['name'],
                    'coords': (ep['x1'], ep['y1'], ep['x2'], ep['y2']),
                    'color': ep['color']
                })
        
        return boundaries

# Global configuration instance
store_config = StoreConfig()

# Save default configuration
if __name__ == "__main__":
    store_config.save_to_file()
    
    # Validate configuration
    issues = store_config.validate_zones()
    if issues:
        print("⚠️ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration is valid")
