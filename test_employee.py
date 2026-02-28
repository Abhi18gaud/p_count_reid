#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from employee_exclusion import EmployeeExclusionSystem
    print("✅ Import successful")
    
    # Test basic functionality
    exclusion_system = EmployeeExclusionSystem()
    print("✅ EmployeeExclusionSystem created")
    
    # Test get_employee_name method
    name = exclusion_system.get_employee_name(1)
    print(f"✅ get_employee_name(1) = {name}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
