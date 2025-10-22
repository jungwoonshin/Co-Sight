# GAIA Benchmark Integration

This folder contains all GAIA benchmark evaluation tools and scripts for CoSight.

## 📁 Files Overview

### Core Evaluation Scripts
- **`gaia_evaluation.py`** - Basic evaluation script for simple testing
- **`advanced_gaia_evaluator.py`** - Advanced evaluation with parallel processing
- **`gaia_results_processor.py`** - Results analysis and visualization tools

### Setup and Utility Scripts
- **`gaia_quick_start.py`** - Interactive quick start script
- **`download_gaia_dataset.py`** - Dataset downloader from Hugging Face
- **`install_gaia_dependencies.py`** - Dependency installer
- **`setup_gaia.py`** - Complete automated setup script

### Documentation
- **`GAIA_INTEGRATION_GUIDE.md`** - Comprehensive integration guide
- **`GAIA_QUICK_START.md`** - Quick start guide

## 🚀 Quick Start

### Option 1: Complete Setup (Recommended)
```bash
python gaia/setup_gaia.py
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
python gaia/install_gaia_dependencies.py --install

# 2. Download dataset
python gaia/download_gaia_dataset.py

# 3. Run evaluation
python gaia/gaia_quick_start.py
```

### Option 3: Direct Evaluation
```bash
# Basic evaluation
python gaia/gaia_evaluation.py --benchmark_path ./gaia_dataset --max_cases 10

# Advanced evaluation
python gaia/advanced_gaia_evaluator.py --benchmark_path ./gaia_dataset --max_cases 50 --max_workers 3
```

## 📊 Results Analysis

After running evaluation, analyze your results:

```bash
python gaia/gaia_results_processor.py \
    --results_file ./gaia_results/gaia_results_detailed_*.json \
    --output_dir ./gaia_analysis
```

## 🔧 Prerequisites

1. **Python 3.11+**
2. **CoSight properly configured** (`.env` file with API keys)
3. **Internet connection** for dataset download
4. **Hugging Face account** (free registration required)

## 📚 Documentation

- **`GAIA_INTEGRATION_GUIDE.md`** - Complete guide with examples and troubleshooting
- **`GAIA_QUICK_START.md`** - Quick start instructions

## 🎯 Features

- **Parallel Processing** - Run multiple test cases simultaneously
- **Automatic Retry** - Retry failed cases automatically
- **Comprehensive Analysis** - Statistical analysis with visualizations
- **Multiple Export Formats** - JSON, CSV, and Markdown reports
- **Interactive Setup** - Guided setup process
- **Error Handling** - Robust error handling and recovery

## 📈 Generated Outputs

- **Detailed Results** - Complete test case information
- **Summary Statistics** - Key performance metrics
- **Visualizations** - Charts and graphs for analysis
- **Comprehensive Reports** - Markdown reports with recommendations
- **CSV Exports** - Data for spreadsheet analysis

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section in the documentation
2. Review the CoSight main documentation
3. Check the GAIA benchmark official documentation
4. Open an issue on the GitHub repository

## 📄 License

This GAIA integration follows the same license as the CoSight project.

