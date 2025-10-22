#!/usr/bin/env python3
"""
GAIA Dataset Downloader

This script helps you download the GAIA benchmark dataset from Hugging Face.
GAIA (General AI Assistant) is a benchmark for evaluating AI assistants on real-world tasks.
"""

import os
import sys
import argparse
from pathlib import Path
import requests
import json


def download_gaia_dataset(output_dir: str = "./gaia_dataset", subset: str = "test"):
    """
    Download GAIA dataset from Hugging Face
    
    Args:
        output_dir: Directory to save the dataset
        subset: Dataset subset to download ('test', 'validation', or 'all')
    """
    
    print("🚀 GAIA Dataset Downloader")
    print("=" * 50)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_path.absolute()}")
    
    # GAIA dataset information
    dataset_info = {
        "name": "GAIA Benchmark",
        "description": "General AI Assistant benchmark for evaluating LLMs with augmented capabilities",
        "url": "https://huggingface.co/datasets/gaia-benchmark/GAIA",
        "github_url": "https://github.com/TattaBio/gaia-benchmark",
        "files": {
            "test": "test.json",
            "validation": "validation.json", 
            "metadata": "metadata.json"
        }
    }
    
    print(f"\n📊 Dataset: {dataset_info['name']}")
    print(f"📝 Description: {dataset_info['description']}")
    print(f"🔗 Repository: {dataset_info['url']}")
    
    print(f"\n⚠️  Important Notes:")
    print(f"   - You need to accept the dataset access conditions on Hugging Face")
    print(f"   - The dataset contains over 450 non-trivial questions")
    print(f"   - It's designed to evaluate LLMs with tooling capabilities")
    
    # Check if user wants to proceed
    proceed = input(f"\n🤔 Do you want to proceed with downloading? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("❌ Download cancelled.")
        return False
    
    print(f"\n📥 Downloading GAIA dataset...")
    
    # Method 1: Try using huggingface_hub if available
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        print("✅ Using huggingface_hub library")
        
        repo_id = "gaia-benchmark/GAIA"
        
        # List available files
        try:
            files = list_repo_files(repo_id)
            print(f"📋 Available files: {files}")
        except Exception as e:
            print(f"⚠️  Could not list files: {e}")
            files = ["test.json", "validation.json", "metadata.json"]
        
        # Download files based on subset
        downloaded_files = []
        
        if subset == "all" or subset == "test":
            try:
                test_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="test.json",
                    local_dir=output_path
                )
                downloaded_files.append(test_file)
                print(f"✅ Downloaded: test.json")
            except Exception as e:
                print(f"❌ Failed to download test.json: {e}")
        
        if subset == "all" or subset == "validation":
            try:
                validation_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="validation.json",
                    local_dir=output_path
                )
                downloaded_files.append(validation_file)
                print(f"✅ Downloaded: validation.json")
            except Exception as e:
                print(f"❌ Failed to download validation.json: {e}")
        
        # Download metadata
        try:
            metadata_file = hf_hub_download(
                repo_id=repo_id,
                filename="metadata.json",
                local_dir=output_path
            )
            downloaded_files.append(metadata_file)
            print(f"✅ Downloaded: metadata.json")
        except Exception as e:
            print(f"⚠️  Could not download metadata.json: {e}")
        
        if downloaded_files:
            print(f"\n🎉 Successfully downloaded {len(downloaded_files)} files!")
            return True
        else:
            print(f"\n❌ No files were downloaded successfully.")
            return False
            
    except ImportError:
        print("❌ huggingface_hub library not found.")
        print("📦 Installing huggingface_hub...")
        
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            print("✅ huggingface_hub installed successfully!")
            print("🔄 Please run the script again.")
            return False
        except Exception as e:
            print(f"❌ Failed to install huggingface_hub: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Error downloading with huggingface_hub: {e}")
        return False


def manual_download_instructions():
    """Provide manual download instructions"""
    print("\n📋 Manual Download Instructions:")
    print("=" * 50)
    
    print("1. 🌐 Visit the GAIA dataset page:")
    print("   https://huggingface.co/datasets/gaia-benchmark/GAIA")
    
    print("\n2. 🔐 Sign up/Login to Hugging Face:")
    print("   - Create a free account if you don't have one")
    print("   - Log in to your account")
    
    print("\n3. ✅ Accept the access conditions:")
    print("   - Read and accept the dataset access conditions")
    print("   - This is required to prevent data leakage")
    
    print("\n4. 📥 Download the files:")
    print("   - Click on 'Files and versions' tab")
    print("   - Download 'test.json' for test cases")
    print("   - Download 'validation.json' for validation cases")
    print("   - Download 'metadata.json' for dataset information")
    
    print("\n5. 📁 Save files to your dataset directory:")
    print("   - Create a directory: ./gaia_dataset")
    print("   - Place the JSON files in this directory")
    
    print("\n6. 🔍 Verify the download:")
    print("   - Check that the files are properly downloaded")
    print("   - Ensure JSON files are valid")
    
    print("\n7. 🔄 Alternative: Use GitHub repository:")
    print("   - Clone: git clone https://github.com/TattaBio/gaia-benchmark.git")
    print("   - The dataset files are in the repository")
    
    print("\n8. 🎯 Quick test with sample data:")
    print("   - Run: python gaia/download_gaia_github.py --sample")
    print("   - This creates a small sample dataset for testing")


def verify_dataset(dataset_path: str):
    """Verify that the downloaded dataset is valid"""
    print(f"\n🔍 Verifying dataset at: {dataset_path}")
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory does not exist: {dataset_path}")
        return False
    
    # Check for required files
    required_files = ["test.json"]
    optional_files = ["validation.json", "metadata.json"]
    
    found_files = []
    missing_files = []
    
    for file in required_files:
        file_path = dataset_path / file
        if file_path.exists():
            found_files.append(file)
            # Check if it's valid JSON
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ {file}: {len(data)} items")
            except Exception as e:
                print(f"⚠️  {file}: Invalid JSON - {e}")
        else:
            missing_files.append(file)
    
    for file in optional_files:
        file_path = dataset_path / file
        if file_path.exists():
            found_files.append(file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ {file}: {len(data)} items")
            except Exception as e:
                print(f"⚠️  {file}: Invalid JSON - {e}")
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print(f"✅ Dataset verification successful!")
    print(f"📊 Found {len(found_files)} files")
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download GAIA benchmark dataset')
    parser.add_argument('--output_dir', default='./gaia_dataset', 
                       help='Directory to save the dataset (default: ./gaia_dataset)')
    parser.add_argument('--subset', choices=['test', 'validation', 'all'], default='test',
                       help='Dataset subset to download (default: test)')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify existing dataset without downloading')
    parser.add_argument('--manual', action='store_true',
                       help='Show manual download instructions only')
    
    args = parser.parse_args()
    
    if args.manual:
        manual_download_instructions()
        return 0
    
    if args.verify_only:
        success = verify_dataset(args.output_dir)
        return 0 if success else 1
    
    # Download dataset
    success = download_gaia_dataset(args.output_dir, args.subset)
    
    if success:
        # Verify the downloaded dataset
        verify_dataset(args.output_dir)
        
        print(f"\n🎉 GAIA dataset download completed!")
        print(f"📁 Dataset location: {Path(args.output_dir).absolute()}")
        print(f"\n🚀 Next steps:")
        print(f"   1. Run GAIA evaluation: python gaia_quick_start.py")
        print(f"   2. Or use advanced evaluation: python advanced_gaia_evaluator.py --benchmark_path {args.output_dir}")
        
        return 0
    else:
        print(f"\n❌ Download failed. Try manual download:")
        manual_download_instructions()
        return 1


if __name__ == "__main__":
    exit(main())
