#!/usr/bin/env python3
"""
GAIA Benchmark Setup Script

This script sets up everything needed to run GAIA benchmark evaluation with CoSight.
It handles dependency installation, dataset download, and initial configuration.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print setup banner"""
    print("🚀 GAIA Benchmark Setup for CoSight")
    print("=" * 50)
    print("This script will set up everything needed to run GAIA evaluation.")
    print()


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def check_env_file():
    """Check if .env file exists and is configured"""
    print("\n🔧 Checking environment configuration...")
    
    env_file = Path(".env")
    env_template = Path(".env_template")
    
    if not env_file.exists():
        if env_template.exists():
            print("⚠️  .env file not found, but .env_template exists")
            print("💡 Please copy .env_template to .env and configure it:")
            print("   cp .env_template .env")
            print("   # Then edit .env with your API keys")
            return False
        else:
            print("❌ Neither .env nor .env_template found")
            return False
    
    print("✅ .env file found")
    
    # Check if .env has been configured (not just template values)
    with open(env_file, 'r') as f:
        content = f.read()
        if "如：" in content or "example" in content.lower():
            print("⚠️  .env file contains template values")
            print("💡 Please configure your API keys in .env file")
            return False
    
    print("✅ .env file appears to be configured")
    return True


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Run the dependency installer
        result = subprocess.run([
            sys.executable, "gaia/install_gaia_dependencies.py", "--install"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def download_dataset():
    """Download GAIA dataset"""
    print("\n📥 Downloading GAIA dataset...")
    
    # Ask user for dataset location
    dataset_path = input("Enter dataset directory (default: ./gaia_dataset): ").strip()
    if not dataset_path:
        dataset_path = "./gaia_dataset"
    
    try:
        # Try Hugging Face download first
        print("📥 Attempting Hugging Face download...")
        result = subprocess.run([
            sys.executable, "gaia/download_gaia_dataset.py", 
            "--output_dir", dataset_path,
            "--subset", "test"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ GAIA dataset downloaded successfully from Hugging Face")
            return True, dataset_path
        else:
            print("⚠️  Hugging Face download failed, trying GitHub...")
            
            # Try GitHub download
            result = subprocess.run([
                sys.executable, "gaia/download_gaia_github.py", 
                "--github",
                "--output_dir", dataset_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ GAIA dataset downloaded successfully from GitHub")
                return True, dataset_path
            else:
                print("⚠️  GitHub download failed, creating sample dataset...")
                
                # Create sample dataset as fallback
                result = subprocess.run([
                    sys.executable, "gaia/download_gaia_github.py", 
                    "--sample",
                    "--output_dir", dataset_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Sample dataset created successfully")
                    print("💡 You can download the full dataset manually later")
                    return True, dataset_path
                else:
                    print(f"❌ All download methods failed")
                    return False, None
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False, None


def create_test_script(dataset_path):
    """Create a test script to verify setup"""
    print("\n📝 Creating test script...")
    
    test_script_content = f'''#!/usr/bin/env python3
"""
GAIA Setup Test Script

This script tests if GAIA evaluation setup is working correctly.
"""

import sys
from pathlib import Path

def test_setup():
    """Test if GAIA setup is working"""
    print("🧪 Testing GAIA Setup")
    print("=" * 30)
    
    # Test 1: Check dependencies
    print("1. Testing dependencies...")
    try:
        import pandas, numpy, matplotlib, seaborn
        print("   ✅ All dependencies available")
    except ImportError as e:
        print(f"   ❌ Missing dependency: {{e}}")
        return False
    
    # Test 2: Check dataset
    print("2. Testing dataset...")
    dataset_path = Path("{dataset_path}")
    if dataset_path.exists():
        json_files = list(dataset_path.glob("*.json"))
        if json_files:
            print(f"   ✅ Dataset found: {{len(json_files)}} files")
        else:
            print("   ❌ No JSON files in dataset directory")
            return False
    else:
        print("   ❌ Dataset directory not found")
        return False
    
    # Test 3: Check CoSight imports
    print("3. Testing CoSight imports...")
    try:
        from CoSight import CoSight
        from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision
        print("   ✅ CoSight imports successful")
    except ImportError as e:
        print(f"   ❌ CoSight import failed: {{e}}")
        return False
    
    print("\\n🎉 All tests passed! GAIA setup is ready.")
    print("\\n🚀 Next steps:")
    print("   python gaia_quick_start.py")
    print("   # or")
    print("   python advanced_gaia_evaluator.py --benchmark_path {dataset_path} --max_cases 5")
    
    return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
'''
    
    with open("test_gaia_setup.py", "w") as f:
        f.write(test_script_content)
    
    print("✅ Test script created: test_gaia_setup.py")
    return True


def main():
    """Main setup function"""
    print_banner()
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return 1
    
    # Step 2: Check environment configuration
    env_ok = check_env_file()
    if not env_ok:
        print("\n⚠️  Please configure your .env file before continuing")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed not in ['y', 'yes']:
            print("❌ Setup cancelled")
            return 1
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        return 1
    
    # Step 4: Download dataset
    dataset_success, dataset_path = download_dataset()
    if not dataset_success:
        print("\n⚠️  Dataset download failed, but you can download manually later")
        dataset_path = "./gaia_dataset"  # Default path
    
    # Step 5: Create test script
    create_test_script(dataset_path)
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 GAIA Setup Complete!")
    print("=" * 50)
    
    print(f"\n📁 Files created:")
    print(f"   - gaia/gaia_evaluation.py (basic evaluation)")
    print(f"   - gaia/advanced_gaia_evaluator.py (advanced evaluation)")
    print(f"   - gaia/gaia_results_processor.py (results analysis)")
    print(f"   - gaia/gaia_quick_start.py (interactive quick start)")
    print(f"   - gaia/download_gaia_dataset.py (dataset downloader)")
    print(f"   - gaia/install_gaia_dependencies.py (dependency installer)")
    print(f"   - gaia/setup_gaia.py (complete setup)")
    
    print(f"\n📚 Documentation:")
    print(f"   - gaia/GAIA_INTEGRATION_GUIDE.md (comprehensive guide)")
    print(f"   - gaia/GAIA_QUICK_START.md (quick start guide)")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Quick evaluation: python gaia/gaia_quick_start.py")
    print(f"   2. Advanced evaluation: python gaia/advanced_gaia_evaluator.py --benchmark_path {dataset_path}")
    print(f"   3. Basic evaluation: python gaia/gaia_evaluation.py --benchmark_path {dataset_path}")
    
    print(f"\n💡 Tips:")
    print(f"   - Start with a small number of test cases (--max_cases 5)")
    print(f"   - Check the documentation for detailed usage instructions")
    print(f"   - Use parallel processing for faster evaluation")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        exit(1)
