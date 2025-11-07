#!/usr/bin/env python3
"""
Simple script to verify your Kaggle credentials are set up correctly.
Run this: python verify_kaggle_setup.py
"""

import os
from pathlib import Path

def check_kaggle_setup():
    print("=" * 60)
    print("Checking Kaggle Setup...")
    print("=" * 60)
    
    # Check if kaggle.json exists
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    print(f"\n1. Checking for .kaggle directory...")
    if kaggle_dir.exists():
        print(f"   ✅ Found: {kaggle_dir}")
    else:
        print(f"   ❌ NOT FOUND: {kaggle_dir}")
        print(f"   → Create this folder and put kaggle.json in it")
        return False
    
    print(f"\n2. Checking for kaggle.json file...")
    if kaggle_json.exists():
        print(f"   ✅ Found: {kaggle_json}")
        
        # Check file permissions
        stat_info = os.stat(kaggle_json)
        mode = oct(stat_info.st_mode)[-3:]
        print(f"   File permissions: {mode}")
        
        if mode != "600":
            print(f"   ⚠️  Warning: Permissions should be 600 (read/write for owner only)")
            print(f"   → Right-click kaggle.json → Get Info → Change permissions")
        else:
            print(f"   ✅ Permissions are correct")
        
        # Try to read the file
        try:
            with open(kaggle_json, 'r') as f:
                content = f.read()
                if 'username' in content and 'key' in content:
                    print(f"   ✅ File looks valid (contains username and key)")
                else:
                    print(f"   ⚠️  File exists but might be empty or invalid")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            return False
        
    else:
        print(f"   ❌ NOT FOUND: {kaggle_json}")
        print(f"   → Move your downloaded kaggle.json file here")
        return False
    
    # Try importing kaggle
    print(f"\n3. Checking if Kaggle API is installed...")
    try:
        import kaggle
        print(f"   ✅ Kaggle API is installed")
    except ImportError:
        print(f"   ❌ Kaggle API not installed")
        print(f"   → Run: pip install kaggle")
        return False
    
    # Try to use the API
    print(f"\n4. Testing Kaggle API connection...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"   ✅ Successfully authenticated with Kaggle!")
        print(f"   ✅ Everything is set up correctly!")
        return True
    except Exception as e:
        print(f"   ❌ Authentication failed: {e}")
        print(f"   → Check that your kaggle.json file is valid")
        return False

if __name__ == "__main__":
    success = check_kaggle_setup()
    print("\n" + "=" * 60)
    if success:
        print("✅ Setup complete! You can now download datasets.")
    else:
        print("❌ Setup incomplete. Follow the instructions above.")
    print("=" * 60)


