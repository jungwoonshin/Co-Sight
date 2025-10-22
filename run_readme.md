# Run GAIA evaluation with Poetry
poetry run python gaia/gaia_quick_start.py

# Run advanced evaluation
poetry run python gaia/advanced_gaia_evaluator.py --benchmark_path ./gaia_dataset --max_cases 1

# Run basic evaluation
poetry run python gaia/gaia_evaluation.py --benchmark_path ./gaia_dataset

# Test setup
poetry run python test_gaia_setup.py