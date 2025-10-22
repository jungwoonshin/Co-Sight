#!/usr/bin/env python3
"""
GAIA Quick Test Script

This script runs a quick GAIA evaluation test with the sample dataset.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gaia.gaia_evaluation import GAIAEvaluator


def run_quick_test():
    """Run a quick GAIA test with sample dataset"""
    print("🚀 GAIA Quick Test")
    print("=" * 30)
    
    # Use the sample dataset
    dataset_path = "./gaia_dataset"
    
    if not Path(dataset_path).exists():
        print("❌ Sample dataset not found!")
        print("💡 Run: python gaia/download_gaia_github.py --sample")
        return False
    
    print(f"📊 Using sample dataset: {dataset_path}")
    print("🎯 Running evaluation with 3 test cases...")
    
    try:
        # Create evaluator
        evaluator = GAIAEvaluator(
            benchmark_path=dataset_path,
            output_dir="./gaia_test_results"
        )
        
        # Run evaluation with 3 test cases
        metrics = evaluator.run_evaluation(max_cases=3)
        
        print("\n🎉 Test completed successfully!")
        print(f"📈 Results:")
        print(f"   - Total cases: {metrics['total_cases']}")
        print(f"   - Correct answers: {metrics['correct_cases']}")
        print(f"   - Accuracy: {metrics['accuracy']:.2%}")
        print(f"   - Average execution time: {metrics['avg_execution_time']:.2f}s")
        
        print(f"\n📁 Results saved to: ./gaia_test_results")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)

