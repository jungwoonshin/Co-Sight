#!/usr/bin/env python3
"""
Install GAIA Evaluation Dependencies

This script installs all required dependencies for GAIA benchmark evaluation.
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install required dependencies for GAIA evaluation"""
    
    print("🔧 Installing GAIA Evaluation Dependencies")
    print("=" * 50)
    
    # Required packages for GAIA evaluation
    packages = [
        "pandas",           # Data manipulation and analysis
        "numpy",            # Numerical computing
        "matplotlib",       # Plotting and visualization
        "seaborn",          # Statistical data visualization
        "huggingface_hub",  # For downloading datasets from Hugging Face
        "requests",         # HTTP library for API calls
    ]
    
    print("📦 Installing packages:")
    for package in packages:
        print(f"   - {package}")
    
    print(f"\n🚀 Installing {len(packages)} packages...")
    
    try:
        # Install packages using pip
        for package in packages:
            print(f"📥 Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}: {result.stderr}")
                return False
        
        print(f"\n🎉 All dependencies installed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    
    print("🔍 Checking GAIA Evaluation Dependencies")
    print("=" * 50)
    
    packages = [
        "pandas", "numpy", "matplotlib", "seaborn", 
        "huggingface_hub", "requests"
    ]
    
    missing_packages = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print(f"Run: python install_gaia_dependencies.py --install")
        return False
    else:
        print(f"\n🎉 All dependencies are available!")
        return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Install GAIA evaluation dependencies')
    parser.add_argument('--install', action='store_true',
                       help='Install missing dependencies')
    parser.add_argument('--check', action='store_true',
                       help='Check if dependencies are installed')
    
    args = parser.parse_args()
    
    if args.install:
        success = install_dependencies()
        if success:
            print(f"\n✅ Dependencies installation completed!")
            print(f"🚀 You can now run GAIA evaluation scripts.")
        return 0 if success else 1
    
    elif args.check:
        success = check_dependencies()
        return 0 if success else 1
    
    else:
        # Default: check dependencies
        success = check_dependencies()
        if not success:
            print(f"\n💡 To install missing dependencies, run:")
            print(f"   python install_gaia_dependencies.py --install")
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())
