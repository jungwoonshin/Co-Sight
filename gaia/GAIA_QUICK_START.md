# GAIA Benchmark Integration - Quick Start

## 🚀 How to Run CoSight with GAIA Benchmark

I've created a comprehensive GAIA benchmark integration for CoSight. Here's how to get started:

### 1. **Quick Start (Recommended)**

Run the interactive quick start script:

```bash
python gaia/gaia_quick_start.py
```

This will guide you through:
- Setting up the GAIA dataset path
- Choosing evaluation type (basic or advanced)
- Running a test evaluation
- Analyzing results with visualizations

### 2. **Basic Evaluation**

For a simple sequential evaluation:

```bash
python gaia/gaia_evaluation.py \
    --benchmark_path /path/to/gaia/dataset \
    --output_dir ./gaia_results \
    --max_cases 10
```

### 3. **Advanced Evaluation (Recommended for Full Testing)**

For parallel processing and comprehensive analysis:

```bash
python gaia/advanced_gaia_evaluator.py \
    --benchmark_path /path/to/gaia/dataset \
    --output_dir ./gaia_results \
    --max_cases 50 \
    --max_workers 3 \
    --timeout 300
```

### 4. **Results Analysis**

Analyze your results with detailed visualizations:

```bash
python gaia/gaia_results_processor.py \
    --results_file ./gaia_results/gaia_results_detailed_*.json \
    --output_dir ./gaia_analysis
```

## 📁 Files Created

I've created the following files for GAIA integration:

1. **`gaia/gaia_evaluation.py`** - Basic evaluation script
2. **`gaia/advanced_gaia_evaluator.py`** - Advanced evaluation with parallel processing
3. **`gaia/gaia_results_processor.py`** - Results analysis and visualization
4. **`gaia/gaia_quick_start.py`** - Interactive quick start script
5. **`gaia/download_gaia_dataset.py`** - Dataset downloader
6. **`gaia/install_gaia_dependencies.py`** - Dependency installer
7. **`gaia/setup_gaia.py`** - Complete automated setup
8. **`test_gaia_setup.py`** - Setup verification script
9. **`gaia/GAIA_INTEGRATION_GUIDE.md`** - Comprehensive documentation

## 🔧 Prerequisites

1. **Install additional dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

2. **Download GAIA dataset:**
   - Visit: https://github.com/gaia-benchmark/GAIA
   - Download the test dataset
   - Place it in a directory accessible to the scripts

3. **Configure your environment:**
   - Ensure `.env` file is properly configured
   - Set up LLM API keys and search engine APIs

## 📊 What You'll Get

After running the evaluation, you'll get:

### Results Files
- **Detailed JSON results** with all test case information
- **Summary statistics** with key metrics
- **CSV export** for spreadsheet analysis

### Analysis & Visualizations
- **Accuracy distribution** charts
- **Execution time analysis** plots
- **Confidence score** distributions
- **Correlation analysis** between metrics
- **Error analysis** and retry statistics
- **Comprehensive report** in Markdown format

### Key Metrics
- **Overall accuracy** percentage
- **Success rate** (cases without errors)
- **Average execution time** per case
- **Confidence vs accuracy** correlation
- **Retry statistics** for failed cases

## 🎯 Example Workflow

```bash
# 1. Quick test (5 cases)
python gaia/gaia_quick_start.py

# 2. Full evaluation (50 cases with parallel processing)
python gaia/advanced_gaia_evaluator.py \
    --benchmark_path ./gaia_dataset \
    --max_cases 50 \
    --max_workers 3

# 3. Analyze results
python gaia/gaia_results_processor.py \
    --results_file ./gaia_results/gaia_results_detailed_*.json \
    --output_dir ./analysis
```

## 🔍 Features

### Advanced Evaluation Features
- **Parallel processing** for faster evaluation
- **Automatic retry** for failed cases
- **Confidence scoring** for predictions
- **Comprehensive error handling**
- **Intermediate result saving**

### Analysis Features
- **Statistical analysis** with detailed metrics
- **Visualization generation** (PNG, PDF, SVG)
- **Correlation analysis** between metrics
- **Performance recommendations**
- **Export to multiple formats**

## 📖 Documentation

For detailed information, see:
- **`GAIA_INTEGRATION_GUIDE.md`** - Complete guide with examples
- **Command-line help** - Run any script with `--help`
- **API documentation** - Available in the guide

## 🚨 Troubleshooting

### Common Issues:
1. **Dataset not found** - Check GAIA dataset path
2. **Memory issues** - Reduce `max_workers` parameter
3. **Timeout errors** - Increase `timeout` parameter
4. **API rate limits** - Reduce concurrent workers

### Debug Mode:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎉 Ready to Start!

1. **Download GAIA dataset** from the official repository
2. **Run the quick start script**: `python gaia/gaia_quick_start.py`
3. **Follow the interactive prompts**
4. **Analyze your results** with the generated visualizations

The integration provides everything you need to evaluate CoSight's performance on the GAIA benchmark with comprehensive analysis and reporting!
