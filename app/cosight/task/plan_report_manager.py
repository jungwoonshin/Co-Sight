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

from threading import Lock
from typing import Callable, Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from app.cosight.task.task_manager import TaskManager
from app.cosight.task.todolist import Plan
from app.common.logger_util import logger


class EventManager:
    def __init__(self):
        # Structure: {event_type: {plan_id: [callbacks]}}
        self._subscribers: Dict[str, Dict[str, List[Callable]]] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor()

    def subscribe(self, event_type: str, plan_id: str, callback: Callable):
        """Subscribe to events, associate with plan ID"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = {}
            self._subscribers[event_type].setdefault(plan_id, []).append(callback)
        logger.info(f"Subscribed to {event_type} for plan_id: {plan_id}, total callbacks: {len(self._subscribers[event_type][plan_id])}")

    def publish(self, event_type: str, plan_or_plan_id=None, event_data=None):
        """Publish events - supports Plan objects and tool event data"""
        # Handle special case for tool events
        if event_type == "tool_event" and isinstance(plan_or_plan_id, str) and event_data is not None:
            plan_id = plan_or_plan_id
            callbacks = []
            with self._lock:
                if event_type in self._subscribers and plan_id in self._subscribers[event_type]:
                    callbacks = self._subscribers[event_type][plan_id].copy()

            logger.info(f"Publishing tool_event for plan_id: {plan_id}, callbacks: {len(callbacks)}")
            # For tool events, use synchronous calls to ensure order
            for callback in callbacks:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Tool event callback execution failed: {e}", exc_info=True)
            return
        
        # Original Plan object processing logic
        plan = plan_or_plan_id
        if plan is None:
            return

        # Get plan_id through TaskManager
        plan_id = TaskManager.get_plan_id(plan)
        if not plan_id:
            logger.warning(f"Cannot find plan object ID: {plan}")
            return

        callbacks = []
        with self._lock:
            if event_type in self._subscribers and plan_id in self._subscribers[event_type]:
                callbacks = self._subscribers[event_type][plan_id].copy()

        logger.info(f"Publishing {event_type} for plan_id: {plan_id}, callbacks: {len(callbacks)}")
        for callback in callbacks:
            self._executor.submit(self._safe_callback, callback, plan)

    def unsubscribe(self, event_type: str, plan_id: str, callback: Callable):
        """Unsubscribe from events for specific plan ID"""
        with self._lock:
            if (event_type in self._subscribers and
                    plan_id in self._subscribers[event_type]):
                try:
                    self._subscribers[event_type][plan_id].remove(callback)
                except ValueError:
                    pass

    def _safe_callback(self, callback: Callable, data):
        try:
            callback(data)
        except Exception as e:
            logger.error(f"Callback failed: {str(e)}", exc_info=True)


plan_report_event_manager = EventManager()
