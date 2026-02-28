#!/usr/bin/env python3
"""
Minimal test to debug the exact Re-ID issue
"""

import os
import sys
import numpy as np
import cv2

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_reid_system():
    """Debug the Re-ID system step by step"""
    print("🔍 DEBUG: Re-ID System Analysis")
    print("=" * 50)
    
    try:
        from professional_reid import ProfessionalReIDSystem, GlobalPersonGallery
        
        # Step 1: Check OSNet model
        print("\n📊 Step 1: Checking OSNet Model")
        reid_system = ProfessionalReIDSystem()
        
        if reid_system.model is not None:
            print("✅ OSNet model loaded successfully")
            print(f"   Device: {reid_system.device}")
            print(f"   Model type: {type(reid_system.model)}")
        else:
            print("❌ OSNet model NOT loaded - using fallback")
            return False
        
        # Step 2: Test embedding extraction
        print("\n📊 Step 2: Testing Embedding Extraction")
        
        # Create two different person crops
        person1 = np.random.randint(100, 155, (180, 80, 3), dtype=np.uint8)
        person2 = np.random.randint(50, 100, (180, 80, 3), dtype=np.uint8)
        
        print(f"   Person 1 crop shape: {person1.shape}")
        print(f"   Person 2 crop shape: {person2.shape}")
        
        embedding1 = reid_system.extract_embedding(person1)
        embedding2 = reid_system.extract_embedding(person2)
        
        print(f"   Person 1 embedding shape: {embedding1.shape if embedding1 is not None else 'None'}")
        print(f"   Person 2 embedding shape: {embedding2.shape if embedding2 is not None else 'None'}")
        
        if embedding1 is None or embedding2 is None:
            print("❌ Embedding extraction failed")
            return False
        
        # Step 3: Test similarity calculation
        print("\n📊 Step 3: Testing Similarity Calculation")
        
        # Same person similarity
        same_similarity = reid_system.calculate_similarity(embedding1, embedding1)
        print(f"   Same person similarity: {same_similarity:.6f}")
        
        # Different person similarity
        diff_similarity = reid_system.calculate_similarity(embedding1, embedding2)
        print(f"   Different person similarity: {diff_similarity:.6f}")
        
        # Step 4: Test gallery matching
        print("\n📊 Step 4: Testing Gallery Matching")
        
        # Clear gallery
        gallery_file = "global_person_gallery.pkl"
        if os.path.exists(gallery_file):
            os.remove(gallery_file)
        
        gallery = GlobalPersonGallery(reid_system)
        print(f"   Gallery threshold: {gallery.similarity_threshold}")
        
        # Add person 1
        id1 = gallery.match_or_create_person(person1, "Camera_1", 1, 15)
        print(f"   Person 1 ID: {id1}")
        
        # Try to match person 2 (should be different)
        id2 = gallery.match_or_create_person(person2, "Camera_2", 2, 15)
        print(f"   Person 2 ID: {id2}")
        
        # Try to match person 1 again (should be same)
        id3 = gallery.match_or_create_person(person1, "Camera_2", 3, 15)
        print(f"   Person 1 (again) ID: {id3}")
        
        # Step 5: Analysis
        print("\n📊 Step 5: Analysis")
        print(f"   Same person similarity: {same_similarity:.6f}")
        print(f"   Different person similarity: {diff_similarity:.6f}")
        print(f"   Threshold: {gallery.similarity_threshold}")
        print(f"   Person 1 IDs: {id1} vs {id3} (should be same)")
        print(f"   Person 2 ID: {id2} (should be different)")
        
        # Check results
        if same_similarity > 0.99:
            print("⚠️ WARNING: Same person similarity is too high (might be comparing with itself)")
        
        if diff_similarity > gallery.similarity_threshold:
            print("⚠️ WARNING: Different person similarity is above threshold (false match risk)")
        
        if id1 == id3 and id2 != id1:
            print("✅ SUCCESS: Gallery matching working correctly")
            return True
        else:
            print("❌ FAILED: Gallery matching not working")
            return False
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_reid_system()
    print(f"\n🎯 Debug Result: {'PASSED' if success else 'FAILED'}")
