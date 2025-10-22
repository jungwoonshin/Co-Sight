#!/usr/bin/env python3
"""
GAIA Dataset GitHub Downloader

Alternative download method using GitHub repository since Hugging Face access failed.
"""

import os
import sys
import subprocess
import json
import requests
from pathlib import Path


def download_from_github(output_dir: str = "./gaia_dataset"):
    """Download GAIA dataset from GitHub repository"""
    
    print("🚀 GAIA Dataset GitHub Downloader")
    print("=" * 50)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_path.absolute()}")
    
    # GitHub repository information
    repo_info = {
        "name": "GAIA Benchmark",
        "github_url": "https://github.com/TattaBio/gaia-benchmark",
        "description": "General AI Assistant benchmark for evaluating LLMs with augmented capabilities"
    }
    
    print(f"\n📊 Dataset: {repo_info['name']}")
    print(f"📝 Description: {repo_info['description']}")
    print(f"🔗 GitHub Repository: {repo_info['github_url']}")
    
    print(f"\n⚠️  Note: This will clone the entire repository")
    
    # Check if user wants to proceed
    proceed = input(f"\n🤔 Do you want to proceed with GitHub download? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("❌ Download cancelled.")
        return False
    
    try:
        print(f"\n📥 Cloning GAIA repository...")
        
        # Clone the repository
        clone_cmd = ["git", "clone", repo_info['github_url'], str(output_path / "gaia-benchmark")]
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Repository cloned successfully")
            
            # Look for dataset files in the cloned repository
            repo_path = output_path / "gaia-benchmark"
            dataset_files = []
            
            # Common locations for dataset files
            possible_locations = [
                "data",
                "dataset", 
                "datasets",
                "test_data",
                ".",
                "benchmark_data"
            ]
            
            for location in possible_locations:
                search_path = repo_path / location
                if search_path.exists():
                    json_files = list(search_path.glob("*.json"))
                    if json_files:
                        dataset_files.extend(json_files)
                        print(f"📄 Found JSON files in {location}: {[f.name for f in json_files]}")
            
            if dataset_files:
                print(f"\n✅ Found {len(dataset_files)} dataset files")
                
                # Copy dataset files to main output directory
                for file_path in dataset_files:
                    dest_path = output_path / file_path.name
                    if not dest_path.exists():
                        import shutil
                        shutil.copy2(file_path, dest_path)
                        print(f"📋 Copied: {file_path.name}")
                
                # Clean up cloned repository
                import shutil
                shutil.rmtree(repo_path)
                print("🧹 Cleaned up temporary files")
                
                return True
            else:
                print("❌ No JSON dataset files found in repository")
                print("📋 Repository contents:")
                for item in repo_path.rglob("*"):
                    if item.is_file() and item.suffix in ['.json', '.jsonl', '.csv']:
                        print(f"   - {item.relative_to(repo_path)}")
                
                return False
                
        else:
            print(f"❌ Failed to clone repository: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False


def download_sample_dataset(output_dir: str = "./gaia_dataset"):
    """Create a sample GAIA dataset for testing"""
    
    print("\n🎯 Creating Sample GAIA Dataset")
    print("=" * 40)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample GAIA test cases
    sample_data = [
        {
            "id": "sample_001",
            "Question": "What is the capital of France?",
            "Final answer": "Paris",
            "Level": 1,
            "Type": "factual"
        },
        {
            "id": "sample_002", 
            "Question": "Calculate 15 * 23",
            "Final answer": "345",
            "Level": 1,
            "Type": "mathematical"
        },
        {
            "id": "sample_003",
            "Question": "What is the population of Tokyo?",
            "Final answer": "Approximately 14 million people",
            "Level": 2,
            "Type": "factual"
        },
        {
            "id": "sample_004",
            "Question": "What is the chemical formula for water?",
            "Final answer": "H2O",
            "Level": 1,
            "Type": "scientific"
        },
        {
            "id": "sample_005",
            "Question": "Who wrote 'Romeo and Juliet'?",
            "Final answer": "William Shakespeare",
            "Level": 1,
            "Type": "literary"
        }
    ]
    
    # Save sample dataset
    sample_file = output_path / "test.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created sample dataset: {sample_file}")
    print(f"📊 Sample contains {len(sample_data)} test cases")
    
    # Create metadata
    metadata = {
        "dataset_name": "GAIA Benchmark Sample",
        "description": "Sample dataset for testing GAIA evaluation",
        "version": "1.0",
        "total_cases": len(sample_data),
        "levels": {
            "1": len([case for case in sample_data if case.get("Level") == 1]),
            "2": len([case for case in sample_data if case.get("Level") == 2])
        },
        "note": "This is a sample dataset for testing purposes. Download the full dataset from the official repository."
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created metadata: {metadata_file}")
    
    return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download GAIA dataset from GitHub')
    parser.add_argument('--output_dir', default='./gaia_dataset', 
                       help='Directory to save the dataset (default: ./gaia_dataset)')
    parser.add_argument('--sample', action='store_true',
                       help='Create sample dataset for testing')
    parser.add_argument('--github', action='store_true',
                       help='Download from GitHub repository')
    
    args = parser.parse_args()
    
    if args.sample:
        success = download_sample_dataset(args.output_dir)
        if success:
            print(f"\n🎉 Sample dataset created successfully!")
            print(f"📁 Location: {Path(args.output_dir).absolute()}")
            print(f"\n🚀 You can now test GAIA evaluation:")
            print(f"   python gaia/gaia_quick_start.py")
        return 0 if success else 1
    
    elif args.github:
        success = download_from_github(args.output_dir)
        if success:
            print(f"\n🎉 GAIA dataset downloaded successfully!")
            print(f"📁 Location: {Path(args.output_dir).absolute()}")
            print(f"\n🚀 You can now run GAIA evaluation:")
            print(f"   python gaia/gaia_quick_start.py")
        return 0 if success else 1
    
    else:
        print("🚀 GAIA Dataset Download Options")
        print("=" * 40)
        print("1. Download from GitHub repository")
        print("2. Create sample dataset for testing")
        print("3. Manual download instructions")
        
        choice = input("\nSelect option (1/2/3): ").strip()
        
        if choice == "1":
            success = download_from_github(args.output_dir)
        elif choice == "2":
            success = download_sample_dataset(args.output_dir)
        elif choice == "3":
            print("\n📋 Manual Download Instructions:")
            print("1. Visit: https://github.com/TattaBio/gaia-benchmark")
            print("2. Clone the repository: git clone https://github.com/TattaBio/gaia-benchmark.git")
            print("3. Find dataset files in the repository")
            print("4. Copy JSON files to ./gaia_dataset/")
            return 0
        else:
            print("❌ Invalid choice")
            return 1
        
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())

