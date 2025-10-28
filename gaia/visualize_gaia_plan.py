#!/usr/bin/env python3
"""
Visualize GAIA Planning - Extract and visualize plans from GAIA test cases

This script runs only the planning phase of GAIA evaluation and visualizes the resulting plan.
"""

import sys
import json
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup minimal logging
import logging
logging.basicConfig(level=logging.WARNING)

from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision
from app.cosight.task.task_manager import TaskManager
from app.cosight.task.todolist import Plan
from app.common.logger_util import logger


def visualize_gaia_plan(test_file: str, case_index: int = 0, output_dir: str = "gaia_plan_visualizations"):
    """
    Load a GAIA test case, create a plan, and visualize it
    
    Args:
        test_file: Path to GAIA test JSON file
        case_index: Index of the test case to run (default: 0)
        output_dir: Directory to save visualizations
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test cases
    print(f"Loading test file: {test_file}")
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Get test case
    if isinstance(test_data, list):
        test_cases = test_data
    elif 'test' in test_data:
        test_cases = test_data['test']
    else:
        test_cases = [test_data]
    
    if case_index >= len(test_cases):
        print(f"Error: Case index {case_index} out of range. Available: 0-{len(test_cases)-1}")
        return None
    
    test_case = test_cases[case_index]
    test_id = test_case.get('file_name', test_case.get('id', f'case_{case_index}'))
    question = test_case.get('Question', test_case.get('question', ''))
    
    print(f"\n{'='*80}")
    print(f"GAIA Test Case {case_index}: {test_id}")
    print(f"{'='*80}")
    print(f"Question: {question[:200]}{'...' if len(question) > 200 else ''}")
    print(f"{'='*80}\n")
    
    # Create CoSight instance to generate plan
    from CoSight import CoSight
    
    # Create a workspace directory for this test
    workspace = Path(output_dir) / f"workspace_{test_id}"
    workspace.mkdir(exist_ok=True, parents=True)
    
    # Create CoSight instance
    print("Creating CoSight instance...")
    cosight = CoSight(
        plan_llm=llm_for_plan(),
        act_llm=llm_for_act(),
        tool_llm=llm_for_tool(),
        vision_llm=llm_for_vision(),
        work_space_path=str(workspace),
        message_uuid=f"visualize_{test_id}"
    )
    
    # Extract just the planning part from execute()
    print("Generating plan (this may take a moment)...")
    
    # Simulate the planning phase without execution
    from app.cosight.agent.planner.task_plannr_agent import TaskPlannerAgent
    from app.cosight.agent.planner.instance.planner_agent_instance import create_planner_instance
    
    # Generate the plan
    retry_count = 0
    while not cosight.plan.get_ready_steps() and retry_count < 3:
        print(f"  Creating plan... (attempt {retry_count + 1})")
        create_result = cosight.task_planner_agent.create_plan(question)
        print(f"  Plan creation result: {create_result[:100]}...")
        
        if not cosight.plan.get_ready_steps():
            question += f"\nThe plan creation result is: {create_result}\nCreation failed, please carefully review the plan creation rules and select the create_plan tool to create the plan"
        retry_count += 1
    
    # Get the generated plan
    plan = cosight.plan
    
    print(f"\n{'='*80}")
    print("Generated Plan")
    print(f"{'='*80}")
    
    # Print plan details
    plan.print_dependency_table()
    
    # Visualize
    print("Generating visualization...")
    visualization_path = Path(output_dir) / f"plan_{test_id}.png"
    
    try:
        plan.visualize(output_path=str(visualization_path), title=f"GAIA Test: {test_id}")
        print(f"\n✓ Visualization saved to: {visualization_path}")
    except Exception as e:
        print(f"\n✗ Failed to generate visualization: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Save plan details as JSON
    json_path = Path(output_dir) / f"plan_{test_id}.json"
    plan_data = {
        'test_id': test_id,
        'question': question,
        'title': plan.title,
        'steps': plan.steps,
        'dependencies': plan.dependencies,
        'step_statuses': plan.step_statuses,
        'progress': plan.get_progress()
    }
    
    with open(json_path, 'w') as f:
        json.dump(plan_data, f, indent=2)
    
    print(f"✓ Plan details saved to: {json_path}")
    
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Total steps: {len(plan.steps)}")
    print(f"Dependencies: {len(plan.dependencies)}")
    print(f"Visualization: {visualization_path}")
    print(f"Details: {json_path}")
    print(f"{'='*80}\n")
    
    return plan


def visualize_multiple_cases(test_file: str, num_cases: int = 5, output_dir: str = "gaia_plan_visualizations"):
    """Visualize plans for multiple test cases"""
    
    print(f"Visualizing plans for {num_cases} cases...\n")
    
    for i in range(num_cases):
        try:
            print(f"\n{'#'*80}")
            print(f"Case {i+1}/{num_cases}")
            print(f"{'#'*80}\n")
            
            plan = visualize_gaia_plan(test_file, case_index=i, output_dir=output_dir)
            
            if plan:
                print(f"✓ Successfully visualized case {i+1}")
            else:
                print(f"✗ Failed to visualize case {i+1}")
                
        except Exception as e:
            print(f"\n✗ Error processing case {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("Visualization complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize GAIA test plan generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize first case from test file
  python gaia/visualize_gaia_plan.py gaia_dataset/test.json
  
  # Visualize specific case
  python gaia/visualize_gaia_plan.py gaia_dataset/test.json --case 5
  
  # Visualize first 10 cases
  python gaia/visualize_gaia_plan.py gaia_dataset/test.json --multiple 10
        """
    )
    
    parser.add_argument('test_file', 
                       help='Path to GAIA test JSON file')
    parser.add_argument('--case', '-c', 
                       type=int, default=0,
                       help='Specific case index to visualize (default: 0)')
    parser.add_argument('--multiple', '-m',
                       type=int, default=None,
                       help='Visualize multiple cases (specify count)')
    parser.add_argument('--output', '-o',
                       default='gaia_plan_visualizations',
                       help='Output directory for visualizations (default: gaia_plan_visualizations)')
    
    args = parser.parse_args()
    
    if args.multiple:
        visualize_multiple_cases(args.test_file, args.multiple, args.output)
    else:
        visualize_gaia_plan(args.test_file, args.case, args.output)

