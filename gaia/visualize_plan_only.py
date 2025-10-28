#!/usr/bin/env python3
"""
Simple GAIA Plan Visualizer - Generate and visualize plans without execution

This script creates a plan for a GAIA test case and visualizes it without executing the steps.
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
logging.basicConfig(level=logging.ERROR)

from llm import llm_for_plan
from app.cosight.task.task_manager import TaskManager
from app.cosight.task.todolist import Plan
from app.cosight.agent.planner.task_plannr_agent import TaskPlannerAgent
from app.cosight.agent.planner.instance.planner_agent_instance import create_planner_instance


def create_plan_for_question(question: str, plan_id: str = None):
    """Create a plan for a given question without execution"""
    
    if plan_id is None:
        import time
        plan_id = f"plan_{int(time.time())}"
    
    # Create a new plan
    plan = Plan()
    TaskManager.set_plan(plan_id, plan)
    
    # Create planner agent
    planner_agent = TaskPlannerAgent(
        create_planner_instance("planner_agent"),
        llm_for_plan,  # llm_for_plan is already a ChatLLM object, not a function
        plan_id
    )
    
    # Generate the plan
    print("Generating plan...")
    result = planner_agent.create_plan(question)
    print(f"Plan created: {result[:100]}...")
    
    return plan


def visualize_gaia_case(test_file: str, case_index: int = 0, output_dir: str = "gaia_plan_visualizations"):
    """Load a GAIA case and visualize its plan"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test file
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    if isinstance(test_data, list):
        test_cases = test_data
    elif 'test' in test_data:
        test_cases = test_data['test']
    else:
        test_cases = [test_data]
    
    if case_index >= len(test_cases):
        print(f"Error: Case index {case_index} out of range")
        return None
    
    test_case = test_cases[case_index]
    test_id = test_case.get('file_name', test_case.get('id', f'case_{case_index}'))
    question = test_case.get('Question', test_case.get('question', ''))
    
    print(f"\n{'='*80}")
    print(f"GAIA Test Case {case_index}: {test_id}")
    print(f"{'='*80}")
    print(f"Question: {question}")
    print(f"{'='*80}\n")
    
    # Create plan
    plan = create_plan_for_question(question, f"viz_{test_id}")
    
    if not plan or len(plan.steps) == 0:
        print("✗ Failed to create plan")
        return None
    
    # Print plan
    plan.print_dependency_table()
    
    # Visualize
    viz_path = Path(output_dir) / f"plan_{test_id}.png"
    plan.visualize(output_path=str(viz_path), title=f"GAIA: {test_id}")
    
    print(f"\n✓ Visualization saved to: {viz_path}")
    
    # Save JSON
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
    
    print(f"✓ Plan data saved to: {json_path}\n")
    
    return plan


def visualize_multiple_cases(test_file: str, num_cases: int, output_dir: str = "gaia_plan_visualizations"):
    """Visualize plans for multiple test cases"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test file once
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    if isinstance(test_data, list):
        test_cases = test_data
    elif 'test' in test_data:
        test_cases = test_data['test']
    else:
        test_cases = [test_data]
    
    max_cases = min(num_cases, len(test_cases))
    
    print(f"\n{'='*80}")
    print(f"Visualizing {max_cases} GAIA Test Cases")
    print(f"{'='*80}\n")
    
    successful = 0
    failed = 0
    
    for i in range(max_cases):
        try:
            print(f"\n{'#'*80}")
            print(f"Processing Case {i+1}/{max_cases}")
            print(f"{'#'*80}")
            
            if i >= len(test_cases):
                print(f"Skipping case {i+1}: out of range")
                continue
            
            test_case = test_cases[i]
            test_id = test_case.get('file_name', test_case.get('id', f'case_{i}'))
            question = test_case.get('Question', test_case.get('question', ''))
            
            print(f"\nGAIA Test Case {i}: {test_id}")
            print(f"Question: {question[:200]}{'...' if len(question) > 200 else ''}\n")
            
            # Create plan
            plan = create_plan_for_question(question, f"viz_{test_id}")
            
            if not plan or len(plan.steps) == 0:
                print("✗ Failed to create plan")
                failed += 1
                continue
            
            # Print plan summary
            progress = plan.get_progress()
            print(f"Plan created with {len(plan.steps)} steps")
            print(f"Progress: {progress}")
            
            # Visualize
            viz_path = Path(output_dir) / f"plan_case_{i}_{test_id}.png"
            plan.visualize(output_path=str(viz_path), title=f"GAIA: {test_id}")
            
            print(f"✓ Visualization: {viz_path}")
            
            # Save JSON
            json_path = Path(output_dir) / f"plan_case_{i}_{test_id}.json"
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
            
            print(f"✓ Plan data: {json_path}\n")
            successful += 1
                
        except Exception as e:
            failed += 1
            print(f"✗ Error processing case {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Batch Visualization Complete!")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {successful + failed}")
    print(f"{'='*80}\n")
    
    return successful


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize GAIA test plans')
    parser.add_argument('test_file', nargs='?', default='gaia_dataset/validation.json', help='Path to GAIA test JSON')
    parser.add_argument('--case', '-c', type=int, default=0, help='Case index')
    parser.add_argument('--multiple', '-m', type=int, default=None, help='Visualize multiple cases')
    parser.add_argument('--output', '-o', default='gaia_plan_visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    if args.multiple:
        visualize_multiple_cases(args.test_file, args.multiple, args.output)
    else:
        visualize_gaia_case(args.test_file, args.case, args.output)
