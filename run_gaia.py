#!/usr/bin/env python3
"""
GAIA Evaluator Runner Script
Run this script to evaluate CoSight on GAIA benchmark using Poetry environment
"""

import subprocess
import sys
from pathlib import Path

def run_gaia_evaluation():
    """Run GAIA evaluation using Poetry"""
    print("🚀 Running GAIA Evaluation with Poetry...")
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Build the command
    cmd = [
        "poetry", "run", "python", 
        str(script_dir / "gaia" / "advanced_gaia_evaluator.py"),
        "--benchmark_path", "gaia_dataset",
        "--output_dir", "./gaia_results"
    ]
    
    # Add max_cases if provided as command line argument
    if len(sys.argv) > 1:
        try:
            max_cases = int(sys.argv[1])
            cmd.extend(["--max_cases", str(max_cases)])
            print(f"📊 Running evaluation with max {max_cases} cases")
        except ValueError:
            print("❌ Invalid max_cases argument. Using all cases.")
    
    # Add silent flag if requested
    if "--silent" in sys.argv:
        cmd.append("--silent")
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # Run the command
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print("=" * 50)
        print("✅ GAIA evaluation completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print("=" * 50)
        print(f"❌ GAIA evaluation failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print("=" * 50)
        print(f"❌ Error running GAIA evaluation: {e}")
        return 1

if __name__ == "__main__":
    print("GAIA Benchmark Evaluator for CoSight")
    print("Usage: python run_gaia.py [max_cases] [--silent]")
    print("Example: python run_gaia.py 3 --silent")
    print()
    
    exit_code = run_gaia_evaluation()
    sys.exit(exit_code)


