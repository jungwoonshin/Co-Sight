# GAIA Benchmark Integration with CoSight

This guide provides comprehensive instructions for running CoSight with the GAIA benchmark and analyzing the results.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Running GAIA Evaluation](#running-gaia-evaluation)
5. [Results Analysis](#results-analysis)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

## Overview

GAIA (General AI Assistant) is a comprehensive benchmark for evaluating AI assistants on real-world tasks. This integration allows you to:

- Run CoSight on GAIA benchmark test cases
- Evaluate performance with detailed metrics
- Analyze results with comprehensive visualizations
- Generate detailed reports and recommendations

## Prerequisites

### System Requirements

- Python 3.11 or higher
- CoSight installed and configured
- Required dependencies (see requirements below)

### Dependencies

Install additional dependencies for GAIA evaluation:

```bash
pip install pandas numpy matplotlib seaborn
```

### GAIA Dataset

Download the GAIA benchmark dataset from the official repository:
- [GAIA GitHub Repository](https://github.com/gaia-benchmark/GAIA)
- Place the dataset in a directory accessible to the evaluation scripts

## Setup

### 1. Environment Configuration

Ensure your `.env` file is properly configured with:
- LLM API keys and endpoints
- Search engine API keys (Google, Tavily, etc.)
- Any other required configurations

### 2. Directory Structure

Create the following directory structure:

```
Co-Sight/
├── gaia_evaluation.py          # Basic evaluation script
├── advanced_gaia_evaluator.py  # Advanced evaluation with parallel processing
├── gaia_results_processor.py   # Results analysis and visualization
├── gaia_workspaces/            # Workspace directory (auto-created)
└── gaia_results/              # Results directory (auto-created)
```

## Running GAIA Evaluation

### Basic Evaluation

Run a simple GAIA evaluation:

```bash
python gaia_evaluation.py \
    --benchmark_path /path/to/gaia/dataset \
    --output_dir ./gaia_results \
    --max_cases 10
```

### Advanced Evaluation with Parallel Processing

For better performance and comprehensive analysis:

```bash
python advanced_gaia_evaluator.py \
    --benchmark_path /path/to/gaia/dataset \
    --output_dir ./gaia_results \
    --max_cases 50 \
    --max_workers 3 \
    --timeout 300
```

### Command Line Options

#### Basic Evaluation Script (`gaia_evaluation.py`)

- `--benchmark_path`: Path to GAIA benchmark dataset (required)
- `--output_dir`: Output directory for results (default: `./gaia_results`)
- `--workspace_base`: Base directory for CoSight workspaces (default: `./gaia_workspaces`)
- `--max_cases`: Maximum number of test cases to evaluate
- `--start_index`: Starting index for test cases (default: 0)

#### Advanced Evaluation Script (`advanced_gaia_evaluator.py`)

Additional options:
- `--max_workers`: Maximum number of parallel workers (default: 3)
- `--timeout`: Timeout per test case in seconds (default: 300)
- `--no_retry`: Disable retry of failed cases
- `--max_retries`: Maximum number of retries for failed cases (default: 2)

## Results Analysis

### Automatic Analysis

The evaluation scripts automatically generate:
- Detailed JSON results
- Summary statistics
- CSV export for further analysis
- Comprehensive analysis report (Markdown)

### Manual Analysis

Use the results processor for detailed analysis:

```bash
python gaia_results_processor.py \
    --results_file ./gaia_results/gaia_results_detailed_YYYYMMDD_HHMMSS.json \
    --output_dir ./gaia_analysis \
    --plot_format png
```

### Generated Files

After running evaluation, you'll find:

#### Results Files
- `gaia_results_detailed_*.json`: Complete results with all details
- `gaia_results_summary_*.json`: Summary metrics only
- `gaia_results_*.csv`: CSV format for spreadsheet analysis

#### Analysis Files
- `gaia_analysis_report_*.md`: Comprehensive analysis report
- `gaia_results_processed_*.csv`: Processed data with derived columns
- `gaia_summary_stats_*.json`: Statistical summary

#### Visualizations
- `accuracy_distribution.png`: Accuracy distribution charts
- `execution_time_distribution.png`: Execution time analysis
- `confidence_distribution.png`: Confidence score analysis
- `accuracy_vs_confidence.png`: Correlation analysis
- `execution_time_vs_accuracy.png`: Performance analysis
- `retry_distribution.png`: Retry statistics
- `error_analysis.png`: Error analysis

## Advanced Usage

### Custom Evaluation Logic

You can customize the evaluation by modifying the answer extraction and correctness evaluation methods:

```python
def _extract_answer_from_response(self, response: str, test_case: Dict[str, Any]) -> str:
    """Custom answer extraction logic"""
    # Your custom logic here
    pass

def _evaluate_correctness(self, predicted: str, ground_truth: str) -> bool:
    """Custom correctness evaluation logic"""
    # Your custom logic here
    pass
```

### Parallel Processing Configuration

Adjust parallel processing settings based on your system:

```python
config = GAIAConfig(
    max_workers=4,  # Adjust based on CPU cores
    timeout_per_case=600,  # Increase for complex cases
    retry_failed=True,
    max_retries=3
)
```

### Custom Metrics

Add custom metrics to the evaluation:

```python
def _calculate_custom_metrics(self) -> Dict[str, Any]:
    """Calculate custom evaluation metrics"""
    # Your custom metrics here
    pass
```

## Troubleshooting

### Common Issues

#### 1. Dataset Loading Errors

**Problem**: Cannot find GAIA dataset files
**Solution**: 
- Ensure the dataset path is correct
- Check that JSON/JSONL files exist in the directory
- Verify file permissions

#### 2. Memory Issues

**Problem**: Out of memory during evaluation
**Solution**:
- Reduce `max_workers` parameter
- Process fewer cases at a time (`max_cases`)
- Increase system memory or use swap

#### 3. Timeout Errors

**Problem**: Test cases timing out
**Solution**:
- Increase `timeout` parameter
- Check network connectivity for search APIs
- Verify LLM API responsiveness

#### 4. API Rate Limits

**Problem**: API rate limit exceeded
**Solution**:
- Reduce `max_workers` to decrease concurrent requests
- Add delays between requests
- Use different API keys for different workers

### Debug Mode

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

#### For Large Datasets
- Use parallel processing (`max_workers > 1`)
- Process in batches (`max_cases` parameter)
- Save intermediate results frequently

#### For Complex Cases
- Increase timeout per case
- Enable retry mechanism
- Use more powerful LLM models

## API Reference

### GAIAEvaluator Class

```python
class GAIAEvaluator:
    def __init__(self, benchmark_path: str, output_dir: str, workspace_base: str = None)
    def run_evaluation(self, max_cases: Optional[int] = None, start_index: int = 0) -> Dict[str, Any]
    def evaluate_single_case(self, test_case: Dict[str, Any], case_index: int) -> Dict[str, Any]
```

### AdvancedGAIAEvaluator Class

```python
class AdvancedGAIAEvaluator:
    def __init__(self, config: GAIAConfig)
    def run_evaluation_parallel(self, max_cases: Optional[int] = None, start_index: int = 0) -> Dict[str, Any]
    def evaluate_single_case(self, test_case: Dict[str, Any], case_index: int) -> TestCaseResult
```

### GAIAResultsProcessor Class

```python
class GAIAResultsProcessor:
    def __init__(self, config: AnalysisConfig)
    def generate_summary_statistics(self) -> Dict[str, Any]
    def generate_performance_analysis(self) -> Dict[str, Any]
    def generate_visualizations(self)
    def generate_comprehensive_report(self) -> str
```

### Configuration Classes

```python
@dataclass
class GAIAConfig:
    benchmark_path: str
    output_dir: str
    workspace_base: str = "./gaia_workspaces"
    max_workers: int = 3
    timeout_per_case: int = 300
    save_intermediate: bool = True
    intermediate_interval: int = 10
    detailed_logging: bool = True
    retry_failed: bool = True
    max_retries: int = 2

@dataclass
class AnalysisConfig:
    results_file: str
    output_dir: str
    generate_plots: bool = True
    generate_report: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300
```

## Example Workflows

### Quick Test Run

```bash
# Run 10 test cases for quick evaluation
python gaia_evaluation.py \
    --benchmark_path ./gaia_dataset \
    --max_cases 10 \
    --output_dir ./quick_test
```

### Full Evaluation

```bash
# Run complete evaluation with parallel processing
python advanced_gaia_evaluator.py \
    --benchmark_path ./gaia_dataset \
    --max_workers 4 \
    --timeout 600 \
    --output_dir ./full_evaluation
```

### Results Analysis

```bash
# Analyze results from full evaluation
python gaia_results_processor.py \
    --results_file ./full_evaluation/gaia_results_detailed_*.json \
    --output_dir ./analysis \
    --plot_format pdf
```

### Batch Processing

```bash
# Process dataset in batches
for i in {0..9}; do
    python advanced_gaia_evaluator.py \
        --benchmark_path ./gaia_dataset \
        --start_index $((i * 10)) \
        --max_cases 10 \
        --output_dir ./batch_$i
done
```

## Contributing

To contribute improvements to the GAIA integration:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This GAIA integration follows the same license as the CoSight project.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the CoSight documentation
- Open an issue on the GitHub repository
- Check the GAIA benchmark documentation for dataset-specific questions
