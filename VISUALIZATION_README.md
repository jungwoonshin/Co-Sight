# Plan Dependency Visualization

This tool allows you to visualize the dependency structure of plans in Co-Sight as directed graphs.

## Installation

First, install the required dependencies:

```bash
pip install matplotlib networkx
```

## Usage

### Method 1: Using the Plan object directly

```python
from app.cosight.task.todolist import Plan

# Create a plan with steps and dependencies
plan = Plan(
    title="My Task Plan",
    steps=[
        "Step 1: Initialization",
        "Step 2: Processing", 
        "Step 3: Finalization"
    ],
    dependencies={
        1: [0],  # Step 1 depends on Step 0
        2: [1]   # Step 2 depends on Step 1
    }
)

# Visualize the plan
plan.print_dependency_table()  # Print text table
plan.visualize("my_plan.png")   # Save visualization
```

### Method 2: Command-line visualization script

```bash
# Visualize a specific plan by ID
python visualize_plan.py --plan-id plan_123456 --output plan.png

# Visualize all active plans
python visualize_plan.py --all

# Save all visualizations to a directory
python visualize_plan.py --all --dir visualizations/
```

### Method 3: Example demonstrations

```bash
# Run example visualizations
python example_visualize_plan.py
```

This will create three example visualizations showing different dependency patterns.

## Features

### Visual Elements

- **Node Colors:**
  - ⚪ Gray: Not started
  - 🟡 Gold: In progress
  - 🟢 Green: Completed
  - 🔴 Red: Blocked

- **Arrows:** Show dependencies between steps
- **Layout:** Hierarchical layout using Graphviz (if available)

### Text Output

The `print_dependency_table()` method shows:
- Step index
- Current status
- Dependencies (which steps must complete first)
- Step description

Example output:
```
================================================================================
Plan: Research Project
================================================================================
Step   Status          Dependencies        Step Description
--------------------------------------------------------------------------------
0      not_started     None                Gather initial requirements
1      not_started     0                   Research existing solutions
2      not_started     1                   Design system architecture
...
================================================================================
Progress: {'total': 8, 'completed': 0, 'in_progress': 0, 'blocked': 0, 'not_started': 8}
================================================================================
```

## Understanding Dependencies

### Linear Dependencies

```
Step 0 → Step 1 → Step 2 → Step 3
```

Each step depends only on the previous one.

### Branching Dependencies

```
            Step 1
          ↗       ↘
Step 0 →           → Step 4
          ↘       ↗
            Step 2
```

Multiple steps can run in parallel when they share the same dependencies.

### Complex Dependencies

```
Step 0 → Step 1 → Step 3
              ↘   ↗
                Step 2
```

Some steps may have multiple dependencies and must wait for all of them.

## Example: Real-world Use Case

```python
from app.cosight.task.todolist import Plan

# Web development workflow
plan = Plan(
    title="Web Application Development",
    steps=[
        "Create project repository",
        "Design database schema", 
        "Implement user authentication",
        "Build REST API",
        "Create frontend components",
        "Integration testing",
        "Deploy to production"
    ],
    dependencies={
        1: [0],         # Database design needs repository
        2: [1],         # Auth needs database
        3: [2],         # API needs auth
        4: [1],         # Frontend needs database
        5: [3, 4],      # Testing needs both API and frontend
        6: [5]          # Deployment needs testing
    }
)

# Visualize
plan.visualize("web_dev_workflow.png")
plan.print_dependency_table()
```

This creates a visualization showing:
- Steps 2 and 3 can run in parallel (separate branches)
- Step 4 must wait for both branches to complete
- Final deployment waits for testing

## Integration with Co-Sight Executor

When using Co-Sight to execute plans, you can visualize the current state:

```python
from app.cosight.task.task_manager import TaskManager

# Get your plan
plan_id = "your_plan_id"
plan = TaskManager.get_plan(plan_id)

# Visualize current execution state
plan.visualize(f"execution_state_{plan_id}.png")
plan.print_dependency_table()
```

This shows which steps are:
- Completed (green)
- In progress (gold)
- Blocked (red) 
- Not started (gray)

## Troubleshooting

### Graphviz not available

If you get an error about graphviz, you can still use the tool - it will fall back to a spring layout:

```bash
# Install graphviz for better layouts
brew install graphviz  # macOS
sudo apt-get install graphviz  # Linux
```

### Import errors

Make sure you're running from the project root:

```bash
cd /path/to/Co-Sight
python visualize_plan.py --help
```

### Dependencies not showing

Check that your dependencies are properly formatted as a dictionary:
```python
dependencies = {
    step_index: [list_of_dependency_indices]
}
```

## Advanced Usage

### Custom visualization settings

Modify `visualize_plan.py` to adjust:
- Figure size
- Node colors
- Edge styles
- Layout algorithm

### Batch processing

```python
# Visualize multiple plans
from visualize_plan import visualize_all_plans

visualize_all_plans("my_visualizations/")
```

## File Structure

```
.
├── visualize_plan.py         # Main visualization tool
├── example_visualize_plan.py  # Example demonstrations
├── VISUALIZATION_README.md    # This file
└── app/cosight/task/
    └── todolist.py           # Plan class with .visualize() method
```

