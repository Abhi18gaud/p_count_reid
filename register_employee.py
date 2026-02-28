#!/usr/bin/env python3
"""
Employee Registration Utility
Simple script to register employees for the exclusion system
"""

import cv2
import sys
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from employee_exclusion import EmployeeExclusionSystem

def register_employee_from_image():
    """Register employee from uploaded image"""
    print("👥 Employee Registration from Image")
    print("=" * 50)
    
    # Initialize exclusion system
    exclusion_system = EmployeeExclusionSystem()
    
    # Get employee details
    employee_name = input("Enter employee name: ").strip()
    if not employee_name:
        print("❌ Employee name cannot be empty")
        return False
    
    employee_role = input("Enter employee role (default: Staff): ").strip() or "Staff"
    
    # Create hidden root window for file dialog
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    print("\n📁 Please select an image file...")
    image_path = filedialog.askopenfilename(
        title="Select Employee Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    if not image_path:
        print("❌ No image selected")
        return False
    
    print(f"📸 Selected image: {os.path.basename(image_path)}")
    
    try:
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image. Please check the file format.")
            return False
        
        print(f"📏 Image loaded: {image.shape}")
        
        # Display image for confirmation
        cv2.imshow('Employee Registration - Image Preview', image)
        print("👀 Image preview displayed. Press SPACE to register, ESC to cancel.")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # Register employee
                print("📸 Registering employee from image...")
                
                # Convert BGR to RGB for OSNet
                rgb_image = image[:, :, ::-1]
                success = exclusion_system.register_employee(rgb_image, employee_name, employee_role)
                
                cv2.destroyAllWindows()
                
                if success:
                    print(f"✅ Successfully registered {employee_name} as {employee_role}")
                    print(f"📊 Total registered employees: {len(exclusion_system.employees_db)}")
                    return True
                else:
                    print("❌ Failed to register employee")
                    print("💡 Make sure the face is clearly visible in the image")
                    return False
                    
            elif key == 27:  # ESC
                print("❌ Registration cancelled")
                cv2.destroyAllWindows()
                return False
    
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False

def register_employee_interactive():
    """Interactive employee registration using webcam"""
    print("👥 Employee Registration System")
    print("=" * 50)
    
    # Initialize exclusion system
    exclusion_system = EmployeeExclusionSystem()
    
    # Get employee details
    employee_name = input("Enter employee name: ").strip()
    if not employee_name:
        print("❌ Employee name cannot be empty")
        return False
    
    employee_role = input("Enter employee role (default: Staff): ").strip() or "Staff"
    
    print(f"\n📸 Registering {employee_name} ({employee_role})")
    print("Position the employee's face clearly in front of the camera")
    print("Press SPACE to capture, ESC to cancel")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera. Please check camera connection.")
        return False
    
    print("📹 Camera opened successfully")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read from camera")
            break
        
        # Display instructions
        cv2.putText(frame, "Press SPACE to capture, ESC to cancel", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Registering: {employee_name}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Employee Registration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Capture and register
            print("📸 Capturing face...")
            # Convert BGR to RGB for face_recognition
            rgb_frame = frame[:, :, ::-1]
            success = exclusion_system.register_employee(rgb_frame, employee_name, employee_role)
            
            if success:
                print(f"✅ Successfully registered {employee_name} as {employee_role}")
                print(f"📊 Total registered employees: {len(exclusion_system.employees_db)}")
                return True
            else:
                print("❌ Failed to register employee")
                print("💡 Make sure the face is clearly visible and well-lit")
                return False
                
        elif key == 27:  # ESC
            print("❌ Registration cancelled")
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    return False

def list_employees():
    """List all registered employees"""
    exclusion_system = EmployeeExclusionSystem()
    
    if not exclusion_system.employees_db:
        print("📭 No employees registered")
        return
    
    print("👥 Registered Employees:")
    print("=" * 50)
    
    for emp_id, emp_data in exclusion_system.employees_db.items():
        print(f"ID: {emp_id}")
        print(f"Name: {emp_data['name']}")
        print(f"Role: {emp_data['role']}")
        print(f"Registered: {emp_data.get('registered_at', 'Unknown')}")
        print("-" * 30)

def remove_employee():
    """Remove an employee from the system"""
    exclusion_system = EmployeeExclusionSystem()
    
    if not exclusion_system.employees_db:
        print("📭 No employees registered")
        return
    
    print("👥 Registered Employees:")
    print("=" * 30)
    
    for emp_id, emp_data in exclusion_system.employees_db.items():
        print(f"{emp_id}: {emp_data['name']} ({emp_data['role']})")
    
    try:
        emp_id = int(input("\nEnter employee ID to remove (or 0 to cancel): "))
        if emp_id == 0:
            print("❌ Operation cancelled")
            return
        
        if exclusion_system.remove_employee(emp_id):
            print(f"✅ Employee {emp_id} removed successfully")
        else:
            print(f"❌ Employee {emp_id} not found")
            
    except ValueError:
        print("❌ Invalid employee ID")

def show_active_shifts():
    """Show currently active employee shifts"""
    exclusion_system = EmployeeExclusionSystem()
    active_employees = exclusion_system.get_active_employees()
    
    if not active_employees:
        print("📭 No active employee shifts")
        return
    
    print("⏰ Active Employee Shifts:")
    print("=" * 50)
    
    for emp in active_employees:
        print(f"Name: {emp['name']}")
        print(f"Role: {emp['role']}")
        print(f"Shift Start: {emp['shift_start']}")
        print(f"Last Seen: {emp['last_seen']}")
        print("-" * 30)

def main():
    """Main menu"""
    while True:
        print("\n👥 Employee Management System")
        print("=" * 40)
        print("1. Register New Employee (Webcam)")
        print("2. Register Employee from Image")
        print("3. List All Employees")
        print("4. Remove Employee")
        print("5. Show Active Shifts")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            register_employee_interactive()
        elif choice == '2':
            register_employee_from_image()
        elif choice == '3':
            list_employees()
        elif choice == '4':
            remove_employee()
        elif choice == '5':
            show_active_shifts()
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
