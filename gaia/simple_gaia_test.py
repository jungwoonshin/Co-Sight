#!/usr/bin/env python3
"""
Ultra-simple GAIA test - minimal output
"""

import json
import time
import subprocess
import sys
from pathlib import Path

def run_simple_test():
    """Run a simple GAIA test with minimal output"""
    print("🚀 Running GAIA test...")
    
    # Run the evaluator in background and capture only final result
    cmd = [
        sys.executable, 
        "gaia/advanced_gaia_evaluator.py",
        "--benchmark_path", "./gaia_dataset",
        "--max_cases", "1",
        "--silent"
    ]
    
    try:
        # Run with timeout and capture only the final lines
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        
        if result.returncode == 0:
            # Extract only the final summary lines
            lines = result.stdout.strip().split('\n')
            summary_lines = [line for line in lines if any(keyword in line for keyword in ['🎯', '📊', '⏱️', '❌'])]
            
            if summary_lines:
                print("\n".join(summary_lines))
            else:
                print("✅ Test completed successfully")
        else:
            print(f"❌ Test failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 60 seconds")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_simple_test()
