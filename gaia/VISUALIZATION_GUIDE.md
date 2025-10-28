# GAIA Plan Visualization Guide

This guide shows you how to visualize the planning phase of GAIA test cases without executing the steps.

## Quick Start

### Option 1: Simple Plan Visualization (Recommended)

The simplest way to visualize a GAIA test plan:

```bash
# Visualize the first test case
python gaia/visualize_plan_only.py gaia_dataset/test.json

# Visualize a specific case (e.g., case 5)
python gaia/visualize_plan_only.py gaia_dataset/test.json --case 5

# Specify output directory
python gaia/visualize_plan_only.py gaia_dataset/test.json --case 5 --output my_visualizations/
```

This script:
- ✅ Runs only the planning phase
- ✅ Generates a visualization image
- ✅ Prints a dependency table
- ✅ Saves plan data as JSON
- ✅ Does NOT execute any steps

### Option 2: Full GAIA Workflow Visualization

For more detailed logging and workspace tracking:

```bash
# Visualize first case with full workflow
python gaia/visualize_gaia_plan.py gaia_dataset/test.json

# Visualize multiple cases
python gaia/visualize_gaia_plan.py gaia_dataset/test.json --multiple 10

# Specific case
python gaia/visualize_gaia_plan.py gaia_dataset/test.json --case 3
```

## Installation

Make sure you have the required dependencies:

```bash
pip install matplotlib networkx
```

Optional (for better layouts):
```bash
brew install graphviz  # macOS
sudo apt-get install graphviz  # Linux
```

## Example Output

### 1. Visual Output

A PNG file showing:
- Step nodes colored by status (not started = gray)
- Dependency arrows between steps
- Hierarchical layout
- Step indices and descriptions

**File**: `gaia_plan_visualizations/plan_<test_id>.png`

### 2. Text Output

A dependency table printed to console:

```
================================================================================
Plan: Create comprehensive report on renewable energy
================================================================================
Step   Status          Dependencies        Step Description
--------------------------------------------------------------------------------
0      not_started     None                Research renewable energy sources
1      not_started     0                   Analyze current market trends
2      not_started     1                   Compare different energy types
3      not_started     2                   Create visualizations
4      not_started     3                   Write report
================================================================================
Progress: {'total': 5, 'completed': 0, 'in_progress': 0, 'blocked': 0, 'not_started': 5}
================================================================================
```

### 3. JSON Output

Detailed plan data saved as JSON:

```json
{
  "test_id": "case_001",
  "question": "What is the capital of France?",
  "title": "Research Question Answer",
  "steps": [
    "Search for information about France",
    "Find capital city information",
    "Verify answer"
  ],
  "dependencies": {
    "1": [0],
    "2": [1]
  },
  "step_statuses": {...},
  "progress": {...}
}
```

**File**: `gaia_plan_visualizations/plan_<test_id>.json`

## Understanding the Visualizations

### Node Colors

- ⚪ **Gray**: Not started
- 🟡 **Gold**: In progress
- 🟢 **Green**: Completed
- 🔴 **Red**: Blocked

### Dependency Patterns

#### Linear Dependencies
```
Step 0 → Step 1 → Step 2 → Step 3
```
Each step depends on the previous one.

#### Parallel Steps
```
          Step 1
        ↗        ↘
Step 0 →          → Step 3
        ↘        ↗
          Step 2
```
Steps 1 and 2 can run simultaneously.

#### Complex Dependencies
```
Step 0 → Step 1
      ↘         ↘
        Step 2 → Step 4
      ↗         ↗
Step 3         Step 5
```
Some steps have multiple dependencies.

## Use Cases

### 1. Analyzing Planning Quality

Visualize plans to see if they:
- Break down complex tasks appropriately
- Have reasonable dependencies
- Avoid circular dependencies
- Follow logical flow

### 2. Comparing Different Models

Generate plans from different LLM configurations and compare their strategies:
- Are some models more granular?
- Which create better dependency graphs?
- Do parallel steps improve efficiency?

### 3. Debugging Planning Issues

If plan generation fails:
- Check the visualization for malformed dependencies
- Identify steps that might be causing issues
- See if dependencies are correctly structured

### 4. Understanding GAIA Challenges

Visualize GAIA test cases to understand:
- What planning strategies work for different question types?
- How complex are the solution paths?
- What patterns emerge across different test cases?

## Example: Research Task

```bash
# Visualize a research task
python gaia/visualize_plan_only.py gaia_dataset/test.json --case 10
```

This might generate a plan like:

```
Plan: Research Task on Climate Change
Steps:
  0. Search for recent climate change studies
  1. Collect relevant data from studies
  2. Analyze data trends
  3. Create visualizations
  4. Synthesize findings
  5. Write comprehensive report

Dependencies:
  1 → 0 (Collecting needs searching)
  2 → 1 (Analysis needs data)
  3 → 2 (Visualizations need trends)
  4 → 3 (Synthesis needs visuals)
  5 → 4 (Report needs synthesis)
```

## Batch Processing

Visualize multiple test cases at once:

```bash
# Visualize first 20 cases
python gaia/visualize_gaia_plan.py gaia_dataset/test.json --multiple 20

# Check all outputs in the directory
ls gaia_plan_visualizations/

# View specific visualization
open gaia_plan_visualizations/plan_case_005.png  # macOS
xdg-open gaia_plan_visualizations/plan_case_005.png  # Linux
```

## Integration with Full Execution

After visualizing, if you want to execute:

```bash
# First visualize to see the plan
python gaia/visualize_plan_only.py gaia_dataset/test.json --case 5

# Then run full execution
python gaia/advanced_gaia_evaluator.py --test-file gaia_dataset/test.json --case 5
```

## Troubleshooting

### Import Errors

Make sure you're running from the project root:

```bash
cd /path/to/Co-Sight
python gaia/visualize_plan_only.py ...
```

### No visualization generated

Check if matplotlib is installed:

```bash
pip install matplotlib networkx
```

### Plan not created

If the plan is empty, try:
- Checking the question format
- Verifying LLM API keys are set
- Running with verbose logging

## Advanced Usage

### Custom visualization settings

Edit `visualize_plan.py` to adjust:
- Figure size
- Node colors
- Layout algorithm
- Font sizes

### Export plan programmatically

```python
from gaia.visualize_plan_only import create_plan_for_question

plan = create_plan_for_question(
    "What is the capital of France?"
)

plan.visualize("my_plan.png")
plan.print_dependency_table()
```

## Files Generated

When you run the visualization, you'll get:

```
gaia_plan_visualizations/
├── workspace_<test_id>/      # Workspace for this test
├── plan_<test_id>.png        # Visualization image
└── plan_<test_id>.json       # Plan data as JSON
```

## Next Steps

1. **Visualize** - Use the tools to see how plans are structured
2. **Analyze** - Look for patterns in successful plans
3. **Compare** - Contrast planning strategies across different cases
4. **Improve** - Use insights to enhance planning quality

## See Also

- `VISUALIZATION_README.md` - General plan visualization guide
- `gaia/advanced_gaia_evaluator.py` - Full GAIA evaluation runner
- `app/cosight/task/todolist.py` - Plan class implementation

