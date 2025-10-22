#!/usr/bin/env python3
"""
GAIA Benchmark Quick Start Example

This script demonstrates how to quickly run GAIA evaluation with CoSight.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gaia.gaia_evaluation import GAIAEvaluator
from gaia.advanced_gaia_evaluator import AdvancedGAIAEvaluator, GAIAConfig


def quick_test():
    """Run a quick test with a few GAIA cases"""
    print("🚀 GAIA Benchmark Quick Start")
    print("=" * 50)
    
    # Check if GAIA dataset exists
    gaia_path = input("Enter path to GAIA dataset: ").strip()
    if not gaia_path or not os.path.exists(gaia_path):
        print("❌ GAIA dataset path not found!")
        print("Please download GAIA dataset from: https://github.com/gaia-benchmark/GAIA")
        return
    
    # Get number of test cases
    try:
        max_cases = int(input("Number of test cases to run (default: 5): ") or "5")
    except ValueError:
        max_cases = 5
    
    # Choose evaluation type
    print("\nChoose evaluation type:")
    print("1. Basic evaluation (sequential)")
    print("2. Advanced evaluation (parallel)")
    
    choice = input("Enter choice (1 or 2, default: 1): ").strip() or "1"
    
    output_dir = "./gaia_quick_test"
    
    try:
        if choice == "1":
            print(f"\n📊 Running basic evaluation with {max_cases} test cases...")
            evaluator = GAIAEvaluator(
                benchmark_path=gaia_path,
                output_dir=output_dir
            )
            metrics = evaluator.run_evaluation(max_cases=max_cases)
            
        else:
            print(f"\n📊 Running advanced evaluation with {max_cases} test cases...")
            config = GAIAConfig(
                benchmark_path=gaia_path,
                output_dir=output_dir,
                max_workers=2,  # Conservative for quick test
                timeout_per_case=180  # 3 minutes per case
            )
            evaluator = AdvancedGAIAEvaluator(config)
            metrics = evaluator.run_evaluation_parallel(max_cases=max_cases)
        
        print("\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
        # Show summary
        print(f"\n📈 Summary:")
        print(f"  - Total cases: {metrics['total_cases']}")
        print(f"  - Correct answers: {metrics['correct_cases']}")
        print(f"  - Accuracy: {metrics['accuracy']:.2%}")
        print(f"  - Average execution time: {metrics['avg_execution_time']:.2f}s")
        
        # Ask if user wants to analyze results
        analyze = input("\n🔍 Analyze results with visualizations? (y/n): ").strip().lower()
        if analyze in ['y', 'yes']:
            analyze_results(output_dir)
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("Please check your configuration and try again.")


def analyze_results(results_dir):
    """Analyze the evaluation results"""
    print("\n🔍 Analyzing results...")
    
    # Find the latest results file
    results_path = Path(results_dir)
    json_files = list(results_path.glob("gaia_results_detailed_*.json"))
    
    if not json_files:
        print("❌ No detailed results file found!")
        return
    
    # Use the latest file
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    
    try:
        from gaia.gaia_results_processor import GAIAResultsProcessor, AnalysisConfig
        
        config = AnalysisConfig(
            results_file=str(latest_file),
            output_dir=str(results_path / "analysis"),
            generate_plots=True,
            generate_report=True
        )
        
        processor = GAIAResultsProcessor(config)
        processor.generate_visualizations()
        report_file = processor.generate_comprehensive_report()
        
        print("✅ Analysis completed!")
        print(f"📊 Visualizations saved to: {results_path / 'analysis'}")
        print(f"📄 Report generated: {report_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


def main():
    """Main function"""
    print("Welcome to GAIA Benchmark Evaluation with CoSight!")
    print("This script will help you run a quick evaluation.\n")
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found!")
        print("Please ensure you have configured your environment variables.")
        print("You can copy .env_template to .env and configure it.\n")
    
    try:
        quick_test()
    except KeyboardInterrupt:
        print("\n\n👋 Evaluation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
