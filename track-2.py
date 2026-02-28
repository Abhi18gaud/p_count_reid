"""
Enhanced Retail Analytics Tracking System
Main tracking script with full retail analytics integration
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
import pickle
import os
import torch
import sys
import threading
from datetime import datetime

# Import our enhanced modules
try:
    from retail_tracker import EnhancedRetailTracker, GlobalPersonRegistry
    from database_manager import RetailDatabaseManager
    from store_config import store_config
    from analytics_dashboard import app
    from employee_exclusion import EmployeeExclusionSystem
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all modules are properly installed")
    sys.exit(1)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
# Multi-camera configuration - add/remove cameras as needed
CAMERA_CONFIGS = [
    # {"id": 0, "name": "Camera_1", "output": "outputs/camera1_multi_tracking.mp4"},
    {"id": "test_video.mp4", "name": "Camera_2", "output": "outputs/camera2_multi_tracking.mp4"},
    # Add more cameras here:
    # {"id": 2, "name": "Camera_3", "output": "outputs/camera3_multi_tracking.mp4"},
]

# For testing with video files, you can use:
# CAMERA_CONFIGS = [
#     {"id": "test_video.mp4", "name": "Camera_1", "output": "outputs/camera1_multi_tracking.mp4"},
#     {"id": "test_video2.mp4", "name": "Camera_2", "output": "outputs/camera2_multi_tracking.mp4"},
# ]

# For single camera mode (original behavior), set SINGLE_CAMERA_MODE = True and specify SINGLE_CAMERA_ID
SINGLE_CAMERA_MODE = False
SINGLE_CAMERA_ID = 0
SINGLE_CAMERA_OUTPUT = "outputs/retail_analytics_output.mp4"

SHOW_DISPLAY = True
RUN_DASHBOARD = False
# if RUN_DASHBOARD==True:
#     app.run(debug=True)
# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'retail_analytics',
    'user': 'postgres',
    'password': 'password',
    'port': '5432'
}

# --------------------------------------------------
# Main Application
# --------------------------------------------------

def setup_database():
    """Setup database and create schema"""
    print("🗄️ Setting up database...")
    
    # Try to create schema, but don't fail if it doesn't work
    if os.path.exists('database_schema.sql'):
        with open('database_schema.sql', 'r') as f:
            schema_sql = f.read()
        
        db_manager = RetailDatabaseManager(DB_CONFIG)
        if db_manager.connect():
            try:
                # Execute schema creation in a transaction
                db_manager.connection.autocommit = False
                cursor = db_manager.connection.cursor()
                
                # Split SQL file into individual statements
                statements = schema_sql.split(';')
                for statement in statements:
                    statement = statement.strip()
                    if statement and not statement.startswith('--'):
                        try:
                            cursor.execute(statement)
                        except Exception as e:
                            # Continue even if some statements fail
                            pass
                
                db_manager.connection.commit()
                print("✅ Database schema created/updated")
            except Exception as e:
                print(f"⚠️ Database setup completed with warnings")
            finally:
                db_manager.disconnect()
    
    print("📊 Database setup completed")
    return True

def run_single_camera_tracking(camera_config, stop_event, global_registry, employee_exclusion):
    """Run tracking for a single camera with error handling"""
    print(f"🎬 Starting tracking for {camera_config['name']} (ID: {camera_config['id']})")
    
    try:
        # Validate video source
        # if isinstance(camera_config['id'], str):
        #     if not os.path.exists(camera_config['id']):
        #         print(f"❌ Video file not found: {camera_config['id']}")
        #         return False
        
        # Initialize tracker for this camera
        tracker = EnhancedRetailTracker(
            video_path=camera_config['id'],
            output_path=camera_config['output'],
            show_display=SHOW_DISPLAY,
            camera_id=camera_config['name'],
            stop_event=stop_event,
            global_registry=global_registry,
            employee_exclusion=employee_exclusion
        )
        
        # Run tracking using the built-in run() method
        tracker.run()
        
        print(f"✅ Tracking completed for {camera_config['name']}")
        print(f"📹 Output saved to: {camera_config['output']}")
        
    except Exception as e:
        print(f"❌ Tracking error for {camera_config['name']}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_multi_camera_tracking():
    """Run tracking for multiple cameras simultaneously"""
    import concurrent.futures
    
    print(f"🎬 Starting Multi-Camera Tracking with {len(CAMERA_CONFIGS)} cameras...")
    
    # Create a shared stop event and global Re-ID system
    stop_event = threading.Event()
    
    # Initialize appropriate global system
    try:
        from professional_reid import ProfessionalReIDSystem, GlobalPersonGallery
        reid_system = ProfessionalReIDSystem()
        global_system = GlobalPersonGallery(reid_system)
        print("🧠 Using Professional Re-ID System")
    except ImportError:
        from retail_tracker import GlobalPersonRegistry
        global_system = GlobalPersonRegistry()
        print("⚠️ Using HSV Fallback System")
    
    # Initialize employee exclusion system
    employee_exclusion = EmployeeExclusionSystem()
    print("👥 Employee Exclusion System initialized")
    
    # Create output directory
    for config in CAMERA_CONFIGS:
        os.makedirs(os.path.dirname(config['output']), exist_ok=True)
    
    # Use ThreadPoolExecutor to run cameras in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CAMERA_CONFIGS)) as executor:
        # Submit all camera tracking tasks
        future_to_camera = {
            executor.submit(run_single_camera_tracking, config, stop_event, global_system, employee_exclusion): config 
            for config in CAMERA_CONFIGS
        }
        
        try:
            # Wait for all to complete or handle errors
            for future in concurrent.futures.as_completed(future_to_camera):
                camera_config = future_to_camera[future]
                try:
                    success = future.result()
                    if success:
                        print(f"✅ {camera_config['name']} completed successfully")
                    else:
                        print(f"❌ {camera_config['name']} failed")
                except Exception as exc:
                    print(f"❌ {camera_config['name']} generated an exception: {exc}")
        except KeyboardInterrupt:
            print("\n🛑 KeyboardInterrupt received. Signaling all cameras to stop...")
            stop_event.set()
            
            # Give cameras time to save their outputs properly
            print("⏳ Waiting for cameras to save outputs...")
            import time
            time.sleep(3)  # Give 3 seconds for graceful shutdown
            
            # Wait for all threads to finish with timeout
            for future in future_to_camera:
                try:
                    future.result(timeout=5)  # Wait up to 5 seconds for each camera
                except Exception:
                    future.cancel()
            
            print("🛑 All cameras have been stopped and outputs saved.")
    
    # Save global system state one final time
    try:
        if hasattr(global_system, 'save_gallery'):
            global_system.save_gallery()
            print("💾 Global gallery state saved.")
        elif hasattr(global_system, 'save_encodings'):
            global_system.save_encodings()
            print("💾 Global registry state saved.")
    except Exception as e:
        print(f"⚠️ Error saving global system: {e}")
    
    print("🎉 All camera tracking completed!")

def run_tracking():
    """Run the tracking system (single or multi-camera)"""
    if SINGLE_CAMERA_MODE:
        # Original single camera behavior
        print("🎬 Starting Single Camera Tracking...")
        camera_config = {
            "id": SINGLE_CAMERA_ID,
            "name": f"Camera_{SINGLE_CAMERA_ID}",
            "output": SINGLE_CAMERA_OUTPUT
        }
        stop_event = threading.Event()
        
        # Initialize appropriate global system
        try:
            from professional_reid import ProfessionalReIDSystem, GlobalPersonGallery
            reid_system = ProfessionalReIDSystem()
            global_registry = GlobalPersonGallery(reid_system)
            print("🧠 Using Professional Re-ID System")
        except ImportError:
            global_registry = GlobalPersonRegistry()
            print("⚠️ Using HSV Fallback System")
        
        # Initialize employee exclusion system
        employee_exclusion = EmployeeExclusionSystem()
        print("👥 Employee Exclusion System initialized")
        
        return run_single_camera_tracking(camera_config, stop_event, global_registry, employee_exclusion)
    else:
        # Multi-camera behavior
        return run_multi_camera_tracking()

def start_dashboard():
    """Run the analytics dashboard"""
    print("🌐 Starting Analytics Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔄 Dashboard auto-refreshes every 30 seconds")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return False
    
    return True

    

def print_system_info():
    """Print system information"""
    print("=" * 60)
    print("🏪 RETAIL ANALYTICS TRACKING SYSTEM")
    print("=" * 60)
    
    if SINGLE_CAMERA_MODE:
        print(f"📹 Mode: Single Camera")
        print(f"📹 Camera: Device {SINGLE_CAMERA_ID}")
        print(f"📤 Output: {SINGLE_CAMERA_OUTPUT}")
    else:
        print(f"📹 Mode: Multi-Camera ({len(CAMERA_CONFIGS)} cameras)")
        for i, config in enumerate(CAMERA_CONFIGS, 1):
            if isinstance(config['id'], str):
                source_type = f"File: {config['id']}"
            else:
                source_type = f"Device: {config['id']}"
            print(f"  📹 Camera {i}: {config['name']} ({source_type})")
            print(f"     Output: {config['output']}")
    
    print(f"🖥️  Display: {'Enabled' if SHOW_DISPLAY else 'Disabled'}")
    print(f"🌐 Dashboard: {'Enabled' if RUN_DASHBOARD else 'Disabled'}")
    print("🔥 Device: {}".format('CUDA' if torch.cuda.is_available() else 'CPU'))
    
    # Print department info
    print("\n🏬 Configured Departments:")
    for dept_id, dept in store_config.departments.items():
        print(f"  - {dept['name']}: ({dept['x1']},{dept['y1']}) to ({dept['x2']},{dept['y2']})")
    
    # Print entry points
    print("\n🚪 Configured Entry Points:")
    for ep_id, ep in store_config.entry_points.items():
        if ep['is_entrance']:
            print(f"  - {ep['name']}: ({ep['x1']},{ep['y1']}) to ({ep['x2']},{ep['y2']})")
    
    print("\n📊 Analytics Features:")
    print("  ✅ Footfall entry counting")
    print("  ✅ Department dwell time tracking")
    print("  ✅ Department visit counting")
    print("  ✅ Unique customer counting per department")
    print("  ✅ First department visited tracking")
    print("  ✅ Attended vs unattended customer tracking")
    print("  ✅ Wait time tracking")
    print("  ✅ Real-time dashboard")
    print("  ✅ Multi-camera simultaneous processing")
    print("  ✅ Duplicate ID prevention")
    print("  ✅ Thread-safe global registry")
    print("=" * 60)

def main():
    """Main application entry point"""
    print_system_info()
    
    # Create output directories for all cameras
    if SINGLE_CAMERA_MODE:
        os.makedirs(os.path.dirname(SINGLE_CAMERA_OUTPUT), exist_ok=True)
    else:
        for config in CAMERA_CONFIGS:
            os.makedirs(os.path.dirname(config['output']), exist_ok=True)
    
    # Validate camera inputs
    if not SINGLE_CAMERA_MODE:
        print("\n📹 Validating camera inputs...")
        for config in CAMERA_CONFIGS:
            if isinstance(config['id'], str):
                if os.path.exists(config['id']):
                    print(f"✅ Video file found: {config['id']}")
                else:
                    print(f"⚠️ Video file not found: {config['id']}")
            else:
                print(f"✅ Camera device: {config['id']}")
    
    # Setup database (commented out to prevent SQL schema errors if DB already exists)
    # setup_database()
    
    try:
        # Run tracking
        if run_tracking():
            print("\n🎉 Tracking completed successfully!")
            
            # Ask user if they want to run dashboard
            response = input("\n🌐 Would you like to start the analytics dashboard? (y/n): ")
            run_dashboard = response.lower() == 'y'
            
            if run_dashboard:
                start_dashboard()
        else:
            print("❌ Tracking failed. Please check the error messages above")
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received. Exiting application.")

if __name__ == "__main__":
    main()
