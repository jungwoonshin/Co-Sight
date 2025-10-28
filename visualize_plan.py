#!/usr/bin/env python3
"""
Plan Dependency Visualizer for Co-Sight

This script visualizes the dependency structure of a Plan object as a directed graph.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.cosight.task.todolist import Plan
from app.cosight.task.task_manager import TaskManager
from app.common.logger_util import logger


class PlanVisualizer:
    """Visualize plan dependencies as a DAG"""
    
    def __init__(self, plan: Plan):
        self.plan = plan
        self.graph = nx.DiGraph()
        self._build_graph()
    
    def _build_graph(self):
        """Build the graph from plan dependencies"""
        # Add all nodes
        for i, step in enumerate(self.plan.steps):
            status = self.plan.step_statuses.get(step, "not_started")
            self.graph.add_node(i, 
                              label=step, 
                              status=status)
        
        # Add edges based on dependencies
        for step_index, deps in self.plan.dependencies.items():
            if isinstance(step_index, int) and step_index < len(self.plan.steps):
                for dep in deps:
                    if isinstance(dep, int) and dep < len(self.plan.steps):
                        self.graph.add_edge(dep, step_index)
    
    def visualize(self, output_path: str = None, title: str = None, 
                  figsize: Tuple[int, int] = (14, 10)):
        """Visualize the plan DAG"""
        
        if title is None:
            title = self.plan.title or "Task Plan"
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{title}\n(Directed Acyclic Graph of Dependencies)", 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Use hierarchical layout for better visualization
        try:
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        except:
            # Fallback to spring layout if graphviz not available
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Color coding based on status
        color_map = {
            'not_started': '#E0E0E0',  # Light gray
            'in_progress': '#FFD700',  # Gold
            'completed': '#90EE90',    # Light green
            'blocked': '#FF6347'        # Tomato
        }
        
        # Draw nodes
        for node in self.graph.nodes():
            status = self.graph.nodes[node]['status']
            color = color_map.get(status, '#E0E0E0')
            
            # Draw node as a box
            x, y = pos[node]
            node_label = self.graph.nodes[node]['label']
            
            # Truncate long labels
            if len(node_label) > 40:
                display_label = node_label[:37] + "..."
            else:
                display_label = node_label
            
            # Draw rectangle for node
            box = mpatches.FancyBboxPatch(
                (x - 0.15, y - 0.1), 0.3, 0.2,
                boxstyle="round,pad=0.02",
                edgecolor='black',
                facecolor=color,
                linewidth=1.5
            )
            ax.add_patch(box)
            
            # Add text
            ax.text(x, y, f"Step {node}\n{display_label}", 
                   ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        
        # Draw edges (dependencies)
        for edge in self.graph.edges():
            source, target = edge
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            
            # Draw arrow
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', 
                                     lw=2, 
                                     color='#333333',
                                     connectionstyle="arc3,rad=0.1"))
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#E0E0E0', label='Not Started'),
            mpatches.Patch(color='#FFD700', label='In Progress'),
            mpatches.Patch(color='#90EE90', label='Completed'),
            mpatches.Patch(color='#FF6347', label='Blocked')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Remove axes
        ax.axis('off')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_dependency_table(self):
        """Print a text-based dependency table"""
        print("\n" + "="*80)
        print(f"Plan: {self.plan.title}")
        print("="*80)
        print(f"{'Step':<6} {'Status':<15} {'Dependencies':<20} {'Step Description':<40}")
        print("-"*80)
        
        for i, step in enumerate(self.plan.steps):
            status = self.plan.step_statuses.get(step, "not_started")
            deps = self.plan.dependencies.get(i, [])
            dep_str = ', '.join(map(str, deps)) if deps else "None"
            
            # Truncate long descriptions
            desc = step[:37] + "..." if len(step) > 40 else step
            
            print(f"{i:<6} {status:<15} {dep_str:<20} {desc:<40}")
        
        print("="*80)
        print(f"Progress: {self.plan.get_progress()}")
        print("="*80 + "\n")


def visualize_plan_by_id(plan_id: str, output_path: str = None):
    """Visualize a plan by its ID"""
    plan = TaskManager.get_plan(plan_id)
    if not plan:
        logger.error(f"Plan with id '{plan_id}' not found")
        return None
    
    visualizer = PlanVisualizer(plan)
    visualizer.print_dependency_table()
    visualizer.visualize(output_path)
    return visualizer


def visualize_all_plans(output_dir: str = "plan_visualizations"):
    """Visualize all active plans"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plans = TaskManager.plans
    logger.info(f"Found {len(plans)} plans to visualize")
    
    for plan_id, plan in plans.items():
        logger.info(f"Visualizing plan: {plan_id}")
        visualizer = PlanVisualizer(plan)
        visualizer.print_dependency_table()
        output_path = os.path.join(output_dir, f"plan_{plan_id}.png")
        visualizer.visualize(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize plan dependencies')
    parser.add_argument('--plan-id', type=str, help='Plan ID to visualize')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--all', action='store_true', help='Visualize all active plans')
    parser.add_argument('--dir', type=str, default='plan_visualizations', 
                       help='Directory for visualizations when using --all')
    
    args = parser.parse_args()
    
    if args.all:
        visualize_all_plans(args.dir)
    elif args.plan_id:
        visualize_plan_by_id(args.plan_id, args.output)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python visualize_plan.py --plan-id plan_123 --output plan.png")
        print("  python visualize_plan.py --all")

