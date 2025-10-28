# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import re
from typing import List, Optional, Dict, Tuple
import os
import platform
from pathlib import PureWindowsPath, PurePosixPath

from app.common.logger_util import logger


# Add global dictionaries at the beginning of the file
folder_files_map: Dict[str, List[str]] = {}
subfolder_files_map: Dict[str, List[str]] = {}


class Plan:
    """Represents a single plan with steps, statuses, and execution details as a DAG."""

    def __init__(self, title: str = "", steps: List[str] = None, dependencies: Dict[int, List[int]] = None, work_space_path: str = ""):
        self.title = title
        self.steps = steps if steps else []
        # Use step content as key to store status, notes and detailed information
        self.step_statuses = {step: "not_started" for step in self.steps}
        self.step_notes = {step: "" for step in self.steps}
        self.step_details = {step: "" for step in self.steps}
        self.step_files = {step: "" for step in self.steps}
        # Store tool call information for each step
        self.step_tool_calls = {step: [] for step in self.steps}
        # Use adjacency list to represent dependencies
        if dependencies:
            self.dependencies = self._normalize_dependencies(dependencies)
        else:
            self.dependencies = {i: [i - 1] for i in range(1, len(self.steps))} if len(self.steps) > 1 else {}
        self.result = ""
        self.work_space_path = work_space_path if work_space_path else os.environ.get("WORKSPACE_PATH") or os.getcwd()

    def set_plan_result(self, plan_result):
        self.result = plan_result

    def get_plan_result(self):
        return self.result

    def get_ready_steps(self) -> List[int]:
        """Get all step indices whose prerequisite dependencies are completed

        Returns:
            List[int]: List of step indices that can be executed immediately (returns all eligible steps)
        """
        logger.debug(f"get_ready_steps dependencies: {self.dependencies}")
        ready_steps = []
        for step_index in range(len(self.steps)):
            # Check if this step can be merged with previous steps
            if self._can_merge_with_previous(step_index):
                continue
                
            # Get all dependencies for this step
            dependencies = self.dependencies.get(step_index, [])

            # Check if all dependencies are completed
            if all(self.step_statuses.get(self.steps[int(dep)]) not  in["not_started","in_progress"]  for dep in dependencies):
                # Check if the step itself has not started
                if self.step_statuses.get(self.steps[step_index]) == "not_started":
                    ready_steps.append(step_index)

        return ready_steps

    def _can_merge_with_previous(self, step_index: int) -> bool:
        """Check if this step can be merged with previous steps to reduce LLM calls"""
        if step_index <= 0:
            return False
        
        current_step = self.steps[step_index]
        previous_step = self.steps[step_index - 1]
        
        # Check if previous step is completed
        if self.step_statuses.get(previous_step) != "completed":
            return False
        
        # Define mergeable step patterns
        mergeable_patterns = [
            # Search-related steps
            ("search", "analyze"),
            ("search", "summarize"),
            ("find", "analyze"),
            ("find", "summarize"),
            ("research", "analyze"),
            ("research", "summarize"),
            
            # File operations
            ("read", "process"),
            ("read", "analyze"),
            ("read", "summarize"),
            ("extract", "process"),
            ("extract", "analyze"),
            
            # Data processing
            ("collect", "analyze"),
            ("collect", "process"),
            ("gather", "analyze"),
            ("gather", "process"),
            
            # Simple operations that can be combined
            ("check", "verify"),
            ("validate", "confirm"),
            ("review", "check"),
        ]
        
        # Check if current and previous steps match mergeable patterns
        current_lower = current_step.lower()
        previous_lower = previous_step.lower()
        
        for pattern1, pattern2 in mergeable_patterns:
            if pattern1 in previous_lower and pattern2 in current_lower:
                logger.info(f"Merging step {step_index} with previous step {step_index - 1}")
                return True
        
        # Check for similar content that can be merged
        if self._are_steps_similar(current_step, previous_step):
            logger.info(f"Merging similar steps {step_index - 1} and {step_index}")
            return True
        
        return False

    def _are_steps_similar(self, step1: str, step2: str) -> bool:
        """Check if two steps are similar enough to be merged"""
        # Simple similarity check based on common keywords
        step1_words = set(step1.lower().split())
        step2_words = set(step2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = step1_words.intersection(step2_words)
        union = step1_words.union(step2_words)
        
        if len(union) == 0:
            return False
        
        similarity = len(intersection) / len(union)
        
        # If similarity is high and both steps are simple operations
        return similarity > 0.6 and len(step1_words) < 10 and len(step2_words) < 10

    def merge_steps(self, step_index1: int, step_index2: int) -> bool:
        """Merge two steps into one"""
        if step_index1 >= step_index2 or step_index2 >= len(self.steps):
            return False
        
        step1 = self.steps[step_index1]
        step2 = self.steps[step_index2]
        
        # Create merged step description
        merged_step = f"{step1} and {step2}"
        
        # Update the plan
        self.steps[step_index1] = merged_step
        
        # Remove the second step
        self.steps.pop(step_index2)
        
        # Update statuses and notes
        if step_index1 in self.step_statuses:
            self.step_statuses[merged_step] = self.step_statuses[step1]
            del self.step_statuses[step1]
        
        if step_index1 in self.step_notes:
            self.step_notes[merged_step] = self.step_notes[step1]
            del self.step_notes[step1]
        
        # Update dependencies
        new_dependencies = {}
        for step_idx, deps in self.dependencies.items():
            if step_idx == step_index2:
                # Skip the removed step
                continue
            elif step_idx > step_index2:
                # Adjust indices for steps after the removed step
                new_dependencies[step_idx - 1] = [dep - 1 if dep > step_index2 else dep for dep in deps]
            else:
                new_dependencies[step_idx] = deps
        
        self.dependencies = new_dependencies
        
        logger.info(f"Successfully merged steps {step_index1} and {step_index2}")
        return True

    def update(self, title: Optional[str] = None, steps: Optional[List[str]] = None,
               dependencies: Optional[Dict[int, List[int]]] = None) -> None:
        """Update the plan with new title, steps, or dependencies while preserving completed steps."""
        if title:
            self.title = title
        if type(steps) == str:
            tmep_str = str(steps)
            steps = tmep_str.split("\n")
        if steps:
            # Preserve all existing steps and their statuses
            new_steps = []
            new_statuses = {}
            new_notes = {}
            new_details = {}
            new_tool_calls = {}

            # First, process all steps in the input order
            for step in steps:
                # If step exists in current steps and is started, preserve it
                if step in self.steps and self.step_statuses.get(step) != "not_started":
                    new_steps.append(step)
                    new_statuses[step] = self.step_statuses.get(step)
                    new_notes[step] = self.step_notes.get(step)
                    new_details[step] = self.step_details.get(step)
                    new_tool_calls[step] = self.step_tool_calls.get(step, [])
                # If step exists in current steps and is not started, preserve as not_started
                elif step in self.steps:
                    new_steps.append(step)
                    new_statuses[step] = "not_started"
                    new_notes[step] = self.step_notes.get(step)
                    new_details[step] = self.step_details.get(step)
                    new_tool_calls[step] = self.step_tool_calls.get(step, [])
                # If step is new, add as not_started
                else:
                    new_steps.append(step)
                    new_statuses[step] = "not_started"
                    new_notes[step] = ""
                    new_details[step] = ""
                    new_tool_calls[step] = []

            self.steps = new_steps
            self.step_statuses = new_statuses
            self.step_notes = new_notes
            self.step_details = new_details
            self.step_tool_calls = new_tool_calls
        logger.info(f"before update dependencies: {self.dependencies}")
        if dependencies:
            self.dependencies.clear()
            dependencies = self._normalize_dependencies(dependencies)
            self.dependencies.update(dependencies)
        else:
            self.dependencies = {i: [i - 1] for i in range(1, len(steps))} if len(steps) > 1 else {}
        logger.info(f"after update dependencies: {self.dependencies}")

    def mark_step(self, step_index: int, step_status: Optional[str] = None, step_notes: Optional[str] = None) -> None:
        """Mark a single step with specific statuses, notes, and details.

        Args:
            step_index (int): Index of the step to update
            step_status (Optional[str]): New status for the step
            step_notes (Optional[str]): Notes for the step
        """
        # Validate step index
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}. Valid indices range from 0 to {len(self.steps) - 1}.")
        logger.info(f"step_index: {step_index}, step_status is {step_status},step_notes is {step_notes}")
        step = self.steps[step_index]

        # Update step status
        if step_status is not None:
            self.step_statuses[step] = step_status

        # Update step notes
        if step_notes is not None:
            step_notes, file_path_info = process_text_with_workspace(step_notes, self.work_space_path)
            self.step_notes[step] = step_notes
            self.step_files[step] = file_path_info

        # Validate status if marking as completed
        if step_status == "completed":
            # Check if all dependencies are completed
            if not all(self.step_statuses[self.steps[int(dep)]] == "completed" for dep in
                       self.dependencies.get(step_index, [])):
                raise ValueError(f"Cannot complete step {step_index} before its dependencies are completed")

    def add_tool_call(self, step_index: int, tool_name: str, tool_args: str, tool_result: str = None) -> None:
        """Add tool call information to a specific step.

        Args:
            step_index (int): Index of the step (-1 for global/MCP tools)
            tool_name (str): Name of the tool called
            tool_args (str): Arguments passed to the tool
            tool_result (str): Result returned by the tool (deprecated, will be ignored)
        """
        # Handle global tools (MCP tools) with step_index = -1
        if step_index == -1:
            # Store global tools under a special key
            global_key = "__global_tools__"
            if global_key not in self.step_tool_calls:
                self.step_tool_calls[global_key] = []
            
            tool_call_info = {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_result": tool_result,
                "timestamp": self._get_current_timestamp()
            }
            self.step_tool_calls[global_key].append(tool_call_info)
            logger.info(f"Added global tool call: {tool_name}")
            return
        
        # Handle step-specific tools
        if step_index < 0 or step_index >= len(self.steps):
            raise ValueError(f"Invalid step_index: {step_index}. Valid indices range from 0 to {len(self.steps) - 1}.")
        
        step = self.steps[step_index]
        tool_call_info = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": tool_result,
            "timestamp": self._get_current_timestamp()
        }
        
        if step not in self.step_tool_calls:
            self.step_tool_calls[step] = []
        self.step_tool_calls[step].append(tool_call_info)
        logger.info(f"Added tool call for step {step_index}: {tool_name}")

    def _get_current_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _normalize_dependencies(self, dependencies: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """Convert potentially 1-based indexed dependencies to 0-based indexing.

        Processing points:
        - Accept string keys from JSON input, convert to int uniformly
        - Determine if overall subtraction by 1 is needed: if keys and their values don't contain 0, and minimum value >= 1, treat as 1-based indexing
        - Only perform conversion when the above conditions are met, otherwise return as-is
        """
        try:
            deps_int: Dict[int, List[int]] = {int(k): v for k, v in dependencies.items()}
        except Exception:
            deps_int = dependencies  # Already int keys
        # Uniformly convert values to int as well
        deps_int = {k: [int(d) for d in v] for k, v in deps_int.items()}
        if not deps_int:
            return deps_int
        keys = list(deps_int.keys())
        values = [d for v in deps_int.values() for d in v]
        # If 0 is already included, consider it 0-based indexing
        if 0 in keys or any(d == 0 for d in values):
            return deps_int
        # If minimum key and all dependency values are >=1, treat as 1-based indexing, subtract 1 overall
        if min(keys) >= 1 and (not values or min(values) >= 1):
            return {k - 1: [d - 1 for d in v] for k, v in deps_int.items()}
        return deps_int


    def get_progress(self) -> Dict[str, int]:
        """Get progress statistics of the plan."""
        return {
            "total": len(self.steps),
            "completed": sum(1 for status in self.step_statuses.values() if status == "completed"),
            "in_progress": sum(1 for status in self.step_statuses.values() if status == "in_progress"),
            "blocked": sum(1 for status in self.step_statuses.values() if status == "blocked"),
            "not_started": sum(1 for status in self.step_statuses.values() if status == "not_started")
        }

    def format(self, with_detail: bool = False) -> str:
        """Format the plan for display."""
        output = f"Plan: {self.title}\n"
        output += "=" * len(output) + "\n\n"

        progress = self.get_progress()
        output += f"Progress: {progress['completed']}/{progress['total']} steps completed "
        if progress['total'] > 0:
            percentage = (progress['completed'] / progress['total']) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"Status: {progress['completed']} completed, {progress['in_progress']} in progress, "
        output += f"{progress['blocked']} blocked, {progress['not_started']} not started\n\n"
        output += "Steps:\n"

        for i, step in enumerate(self.steps):
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
                "blocked": "[!]",
            }.get(self.step_statuses.get(step), "[ ]")

            # Show dependencies
            deps = self.dependencies.get(i, [])
            dep_str = f" (depends on: {', '.join(map(str, deps))})" if deps else ""
            output += f"Step{i} :{status_symbol} {step}{dep_str}\n"
            if self.step_notes.get(step):
                output += f"   Notes: {self.step_notes.get(step)}\nDetails: {self.step_details.get(step)}\n" if with_detail else f"   Notes: {self.step_notes.get(step)}\n"

        return output

    def has_blocked_steps(self) -> bool:
        """Check if there are any blocked steps in the plan.
        
        Returns:
            bool: True if any step is blocked, False otherwise
        """
        return any(status == "blocked" for status in self.step_statuses.values())

    def print_dependency_table(self):
        """Print a text-based dependency table"""
        print("\n" + "="*80)
        print(f"Plan: {self.title}")
        print("="*80)
        print(f"{'Step':<6} {'Status':<15} {'Dependencies':<20} {'Step Description':<40}")
        print("-"*80)
        
        for i, step in enumerate(self.steps):
            status = self.step_statuses.get(step, "not_started")
            deps = self.dependencies.get(i, [])
            dep_str = ', '.join(map(str, deps)) if deps else "None"
            
            # Truncate long descriptions
            desc = step[:37] + "..." if len(step) > 40 else step
            
            print(f"{i:<6} {status:<15} {dep_str:<20} {desc:<40}")
        
        print("="*80)
        progress = self.get_progress()
        print(f"Progress: {progress}")
        print("="*80 + "\n")

    def visualize(self, output_path: str = None, title: str = None):
        """Visualize the plan dependencies as a graph.
        
        Args:
            output_path (str, optional): Path to save the visualization. If None, displays inline.
            title (str, optional): Title for the visualization. Defaults to plan title.
            
        Returns:
            str: Path to saved visualization or None if displayed.
        
        Example:
            >>> plan.visualize("plan_graph.png")
            >>> plan.visualize()  # Display inline
        """
        try:
            from visualize_plan import PlanVisualizer
            
            visualizer = PlanVisualizer(self)
            visualizer.print_dependency_table()
            visualizer.visualize(output_path, title, figsize=(14, 10))
            
            return output_path
        except ImportError as e:
            logger.warning(f"Visualization dependencies not installed: {e}")
            logger.info("To enable visualization, install: pip install matplotlib networkx")
            return None
        except Exception as e:
            logger.error(f"Failed to visualize plan: {e}")
            return None


def get_last_folder_name(workspace_path: str) -> str:
    workspace_path = workspace_path if workspace_path else os.environ.get("WORKSPACE_PATH")
    if not workspace_path or not os.path.exists(workspace_path):
        raise ValueError(f"{workspace_path} workspace path not set.")

    current_os = platform.system()
    if current_os == 'Windows':
        path_obj = PureWindowsPath(workspace_path)
    else:
        path_obj = PurePosixPath(workspace_path)

    return path_obj.name


def extract_and_replace_paths(text: str, folder_name: str, workspace_path: str) -> Tuple[str, List[Dict[str, str]]]:
    # Supported file extensions
    valid_extensions = r"(txt|md|pdf|docx|xlsx|csv|json|xml|html|png|jpg|jpeg|svg|py)"

    # ✅ Linux/macOS style: /xxx/yyy/zzz/file.ext
    # ✅ Windows style: C:\xxx\yyy\file.ext (or UNC network path \\Server\Share\file.ext)
    path_file_pattern = rf'([a-zA-Z]:\\[^\s《》]+?\.{valid_extensions}|/[^\s《》]+?\.{valid_extensions})'

    # ✅ Chinese book title quoted file names (platform independent)
    quoted_file_pattern = rf'《([^《》\s]+?\.{valid_extensions})》'

    result_list: List[Dict[str, str]] = []

    # Initialize file list for this folder (if it doesn't exist)
    if folder_name not in folder_files_map:
        folder_files_map[folder_name] = []

    def replace_path_file(match):
        full_path = match.group(1)
        filename = os.path.basename(full_path.replace("\\", "/"))  # Convert backslashes to forward slashes before extraction
        new_path = f"{folder_name}/{filename}"

        # If filename is not in the folder's list, add it
        # if filename not in folder_files_map[folder_name]:
        #     folder_files_map[folder_name].append(filename)
        #     result_list.append({
        #         "name": filename,
        #         "path": new_path
        #     })
        return new_path

    def replace_quoted_file(match):
        filename = match.group(1)
        new_path = f"{folder_name}/{filename}"

        # If filename is not in the folder's list, add it
        # if filename not in folder_files_map[folder_name]:
        #     folder_files_map[folder_name].append(filename)
        #     result_list.append({
        #         "name": filename,
        #         "path": new_path
        #     })
        return new_path

    new_text = re.sub(path_file_pattern, replace_path_file, text)
    new_text = re.sub(quoted_file_pattern, replace_quoted_file, new_text)

    workspace_path = workspace_path if workspace_path else os.environ.get("WORKSPACE_PATH")
    logger.info(f"extract and replace paths >>>>>>>>>>>>>>>>>>>>>>>>>>>> work_space_path: {workspace_path}")
    # Read all files in workspace directory again
    if workspace_path:
        try:
            # Traverse all files in workspace directory
            for filename in os.listdir(workspace_path):
                # If filename is not in the folder's list, add it
                if filename not in folder_files_map[folder_name]:
                    folder_files_map[folder_name].append(filename)
                    result_list.append({
                        "name": filename,
                        "path": f"{folder_name}/{filename}"
                    })

            # Traverse all subdirectories in workspace directory
            for root, dirs, files in os.walk(workspace_path):
                logger.info(f"root:{root}")
                if root != workspace_path:  # Skip root directory as it's already processed above
                    # Get relative path
                    rel_path = os.path.relpath(root, workspace_path)
                    # Build unique identifier for folder
                    folder_key = f"{folder_name}/{rel_path}"

                    # Initialize file list for this folder (if it doesn't exist)
                    if folder_key not in subfolder_files_map:
                        subfolder_files_map[folder_key] = []
                        logger.info(f"subfolder_files_map: {subfolder_files_map}")

                    for filename in files:
                        # If filename is not in the folder's list, add it
                        if filename not in subfolder_files_map[folder_key]:
                            subfolder_files_map[folder_key].append(filename)
                            # Build complete relative path
                            full_rel_path = f"{folder_name}/{rel_path}/{filename}"
                            result_list.append({
                                "name": filename,
                                "path": full_rel_path
                            })
                            logger.info(f"dirs_result_list:{result_list}")
        except Exception as e:
            logger.error(f"Error reading workspace directory: {str(e)}", exc_info=True)

    return new_text, result_list


def process_text_with_workspace(text: str, work_space_path: str) -> Tuple[str, List[Dict[str, str]]]:
    folder_name = get_last_folder_name(work_space_path)
    return extract_and_replace_paths(text, folder_name, work_space_path)
