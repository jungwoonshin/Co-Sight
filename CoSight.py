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
from datetime import datetime
from app.cosight.agent.actor.instance.actor_agent_instance import create_actor_instance
from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision
from app.cosight.task.plan_report_manager import plan_report_event_manager


import os
import time
from threading import Thread

from app.cosight.agent.actor.task_actor_agent import TaskActorAgent
from app.cosight.agent.planner.instance.planner_agent_instance import create_planner_instance
from app.cosight.agent.planner.task_plannr_agent import TaskPlannerAgent
from app.cosight.task.task_manager import TaskManager
from app.cosight.task.todolist import Plan
from app.cosight.task.time_record_util import time_record
from app.common.logger_util import logger


class CoSight:
    def __init__(self, plan_llm, act_llm, tool_llm, vision_llm, work_space_path: str = None, message_uuid: str|None = None):
        self.work_space_path = work_space_path or os.getenv("WORKSPACE_PATH") or os.getcwd()
        self.plan_id = message_uuid if message_uuid else f"plan_{int(time.time())}"
        self.plan = Plan()
        TaskManager.set_plan(self.plan_id, self.plan)
        self.task_planner_agent = TaskPlannerAgent(create_planner_instance("task_planner_agent"), plan_llm,
                                                   self.plan_id)
        self.act_llm = act_llm  # Store llm for later use
        self.tool_llm = tool_llm
        self.vision_llm = vision_llm

    @time_record
    def execute(self, question, output_format=""):
        create_task = question
        retry_count = 0
        while not self.plan.get_ready_steps() and retry_count < 3:
            create_result = self.task_planner_agent.create_plan(create_task, output_format)
            create_task += f"\nThe plan creation result is: {create_result}\nCreation failed, please carefully review the plan creation rules and select the create_plan tool to create the plan"
            retry_count += 1
        
        # Use continuous monitoring instead of waiting for all steps to complete
        active_threads = {}  # Store active threads {step_index: thread}
        
        while True:
            # Check for new executable steps
            ready_steps = self.plan.get_ready_steps()
            
            # Start new executable steps
            for step_index in ready_steps:
                if step_index not in active_threads:
                    logger.info(f"Starting new step {step_index}")
                    thread = Thread(target=self._execute_single_step, args=(question, step_index))
                    thread.daemon = True
                    thread.start()
                    active_threads[step_index] = thread
            
            # Check completed threads
            completed_steps = []
            for step_index, thread in active_threads.items():
                if not thread.is_alive():
                    completed_steps.append(step_index)
            
            # Remove completed threads
            for step_index in completed_steps:
                del active_threads[step_index]
                logger.info(f"Step {step_index} completed and thread removed")
            
            # Exit if no active threads and no executable steps
            if not active_threads and not ready_steps:
                logger.info("No more ready steps to execute and no active threads")
                break
            
            # Brief sleep to avoid high CPU usage
            import time
            time.sleep(0.1)
        
        return self.task_planner_agent.finalize_plan(question, output_format)

    def _execute_single_step(self, question, step_index):
        """Execute a single step"""
        try:
            logger.info(f"Starting execution of step {step_index}")
            # Each thread creates an independent TaskActorAgent instance
            task_actor_agent = TaskActorAgent(
                create_actor_instance(f"actor_for_step_{step_index}", self.work_space_path),
                self.act_llm,
                self.vision_llm,
                self.tool_llm,
                self.plan_id,
                work_space_path=self.work_space_path
            )
            result = task_actor_agent.act(question=question, step_index=step_index)
            logger.info(f"Completed execution of step {step_index} with result: {result}")
        except Exception as e:
            logger.error(f"Error executing step {step_index}: {e}", exc_info=True)

    def execute_steps(self, question, ready_steps):
        from threading import Thread, Semaphore
        from queue import Queue

        results = {}
        result_queue = Queue()
        semaphore = Semaphore(min(5, len(ready_steps)))

        def execute_step(step_index):
            semaphore.acquire()
            try:
                logger.info(f"Starting execution of step {step_index}")
                # Each thread creates an independent TaskActorAgent instance
                task_actor_agent = TaskActorAgent(
                    create_actor_instance(f"actor_for_step_{step_index}", self.work_space_path),
                    self.act_llm,
                    self.vision_llm,
                    self.tool_llm,
                    self.plan_id,
                    work_space_path=self.work_space_path
                )
                result = task_actor_agent.act(question=question, step_index=step_index)
                logger.info(f"Completed execution of step {step_index} with result: {result}")
                result_queue.put((step_index, result))
            finally:
                semaphore.release()

        # Create and execute threads for each ready_step
        threads = []
        for step_index in ready_steps:
            thread = Thread(target=execute_step, args=(step_index,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results
        while not result_queue.empty():
            step_index, result = result_queue.get()
            results[step_index] = result

        return results


if __name__ == '__main__':
    # Configure workspace
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Get current time and format it
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    # Construct path: /xxx/xxx/work_space/work_space_timestamp
    work_space_path = os.path.join(BASE_DIR, 'work_space', f'work_space_{timestamp}')
    os.makedirs(work_space_path, exist_ok=True)

    # Configure CoSight
    cosight = CoSight(llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision, work_space_path)

    # Run CoSight
    result = cosight.execute("How many wheels are there in a car?")
    logger.info(f"final result is {result}")
