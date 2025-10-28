#!/usr/bin/env python3
"""
Example script demonstrating plan visualization

This script shows how to create a Plan with dependencies and visualize it.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.cosight.task.todolist import Plan
from app.cosight.task.task_manager import TaskManager


def create_example_plan():
    """Create an example plan with dependencies"""
    
    # Example: Research project plan
    steps = [
        "Gather initial requirements",
        "Research existing solutions",
        "Design system architecture", 
        "Implement frontend",
        "Implement backend",
        "Integration testing",
        "Create documentation",
        "Final review and deployment"
    ]
    
    # Define dependencies as a dictionary
    # Key: step index, Value: list of step indices this step depends on
    dependencies = {
        1: [0],      # Research depends on requirements
        2: [1],      # Design depends on research
        3: [2],      # Frontend depends on design
        4: [2],      # Backend depends on design
        5: [3, 4],   # Testing depends on both frontend and backend
        6: [5],      # Documentation depends on testing
        7: [6]       # Final review depends on documentation
    }
    
    plan = Plan(
        title="Research Project with Complex Dependencies",
        steps=steps,
        dependencies=dependencies
    )
    
    return plan


def create_simple_linear_plan():
    """Create a simple linear plan"""
    
    steps = [
        "Step 1: Data collection",
        "Step 2: Data cleaning", 
        "Step 3: Data analysis",
        "Step 4: Generate report"
    ]
    
    # Linear dependencies (each step depends on previous)
    # This is the default behavior if no dependencies specified
    plan = Plan(title="Simple Linear Pipeline", steps=steps)
    
    return plan


def create_branching_plan():
    """Create a plan with branching dependencies"""
    
    steps = [
        "Initialize project",
        "Develop feature A",
        "Develop feature B",
        "Develop feature C", 
        "Integration testing",
        "Deployment"
    ]
    
    dependencies = {
        1: [0],      # Feature A depends on initialization
        2: [0],      # Feature B depends on initialization
        3: [0],      # Feature C depends on initialization
        4: [1, 2, 3],  # Testing depends on all features
        5: [4]       # Deployment depends on testing
    }
    
    plan = Plan(
        title="Branching Development Workflow",
        steps=steps,
        dependencies=dependencies
    )
    
    return plan


def main():
    """Main function to demonstrate plan visualization"""
    
    print("="*80)
    print("Plan Visualization Examples")
    print("="*80)
    
    # Example 1: Simple linear plan
    print("\nExample 1: Simple Linear Plan")
    print("-"*80)
    plan1 = create_simple_linear_plan()
    plan1.visualize("example_simple_plan.png")
    
    # Example 2: Branching plan
    print("\nExample 2: Branching Plan") 
    print("-"*80)
    plan2 = create_branching_plan()
    plan2.visualize("example_branching_plan.png")
    
    # Example 3: Complex dependencies
    print("\nExample 3: Complex Dependencies")
    print("-"*80)
    plan3 = create_example_plan()
    plan3.visualize("example_complex_plan.png")
    
    print("\n" + "="*80)
    print("All visualizations saved!")
    print("Files created:")
    print("  - example_simple_plan.png")
    print("  - example_branching_plan.png")
    print("  - example_complex_plan.png")
    print("="*80)
    
    # Show the dependency table for one plan
    print("\nDetailed dependency table for complex plan:")
    plan3.print_dependency_table()


if __name__ == "__main__":
    main()

