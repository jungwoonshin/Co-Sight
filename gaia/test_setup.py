#!/usr/bin/env python3
"""
Minimal GAIA setup test
"""

import json
import sys
from pathlib import Path

def test_setup():
    """Test basic GAIA setup without running evaluation"""
    print("🔍 Testing GAIA setup...")
    
    # Test 1: Check dataset
    dataset_path = Path("./gaia_dataset")
    if dataset_path.exists():
        json_files = list(dataset_path.glob("*.json"))
        if json_files:
            print(f"✅ Dataset found: {len(json_files)} files")
        else:
            print("❌ No JSON files in dataset")
            return False
    else:
        print("❌ Dataset directory not found")
        return False
    
    # Test 2: Check imports
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from CoSight import CoSight
        from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision
        print("✅ CoSight imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 3: Check sample data
    try:
        # Find the test.json file (not metadata)
        test_file = next((f for f in json_files if 'test' in f.name.lower()), json_files[0])
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Sample data loaded: {len(data)} test cases")
        if len(data) > 0:
            first_case = data[0]
            question = first_case.get('Question', first_case.get('question', 'No question'))
            print(f"   First case: {question[:50]}...")
        
        print("\n🎯 Setup test completed successfully!")
        print("💡 To run evaluation: poetry run python gaia/advanced_gaia_evaluator.py --benchmark_path ./gaia_dataset --max_cases 1")
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False
    

if __name__ == "__main__":
    test_setup()
