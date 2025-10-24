#!/usr/bin/env python3
"""
Simple GAIA Benchmark Runner for CoSight
"""

import json
import time
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress all logging and warnings - BEFORE imports
logging.basicConfig(level=logging.CRITICAL, handlers=[logging.NullHandler()])
for logger_name in ['co-sight', 'httpx', 'httpcore', 'openai', 'lagent']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).handlers = [logging.NullHandler()]
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger().handlers = [logging.NullHandler()]

# Suppress all warnings including Pydantic deprecation warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from CoSight import CoSight
from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision
from app.cosight.llm.chat_llm import ChatLLM


# Global storage for LLM call logs
_llm_call_logs = []
_current_test_id = None
_current_output_dir = None


def log_llm_call(call_type: str, model: str, messages: List[Dict[str, Any]], response: str, metadata: Dict = None):
    """Log an LLM call with all relevant information and write to disk instantly"""
    log_entry = {
        'timestamp': time.time(),
        'call_type': call_type,
        'model': model,
        'messages': messages,
        'response': response,
        'metadata': metadata or {}
    }
    _llm_call_logs.append(log_entry)

    # Instantly write to disk if we have a current test ID
    if _current_test_id and _current_output_dir:
        _write_instant_log(_current_test_id, log_entry, len(_llm_call_logs))


def _write_instant_log(test_id: str, log_entry: Dict[str, Any], call_number: int):
    """Write a single LLM log entry to disk instantly"""
    try:
        log_file = Path(_current_output_dir) / f"llm_logs_{test_id}.json"
        md_file = Path(_current_output_dir) / f"llm_logs_{test_id}.md"

        # Convert log entry to JSON-serializable format
        def make_serializable(obj, max_str_length=50000):
            """Recursively convert objects to JSON-serializable format"""
            if isinstance(obj, dict):
                return {k: make_serializable(v, max_str_length) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item, max_str_length) for item in obj]
            elif isinstance(obj, str):
                # Truncate very long strings to prevent issues
                if len(obj) > max_str_length:
                    return obj[:max_str_length] + f"\n...[TRUNCATED - {len(obj) - max_str_length} more characters]"
                return obj
            elif hasattr(obj, '__dict__'):
                # Convert objects with __dict__ to dict
                return make_serializable(vars(obj), max_str_length)
            elif hasattr(obj, 'model_dump'):
                # Pydantic v2 models
                try:
                    return make_serializable(obj.model_dump(), max_str_length)
                except Exception:
                    return str(obj)[:max_str_length]
            elif hasattr(obj, 'dict'):
                # Pydantic v1 models
                try:
                    return make_serializable(obj.dict(), max_str_length)
                except Exception:
                    return str(obj)[:max_str_length]
            else:
                # For everything else, convert to string
                result = str(obj)
                if len(result) > max_str_length:
                    return result[:max_str_length] + f"\n...[TRUNCATED - {len(result) - max_str_length} more characters]"
                return result

        # Append to JSON file
        formatted_log = {
            'call_number': call_number,
            'timestamp': log_entry['timestamp'],
            'call_type': log_entry['call_type'],
            'model': log_entry['model'],
            'metadata': make_serializable(log_entry['metadata']),
            'messages': make_serializable(log_entry['messages']),
            'response': make_serializable(log_entry['response'])
        }

        # Read existing logs or create new structure
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                # If file is corrupted, start fresh
                data = {
                    'test_id': test_id,
                    'total_calls': 0,
                    'calls': []
                }
        else:
            data = {
                'test_id': test_id,
                'total_calls': 0,
                'calls': []
            }

        data['calls'].append(formatted_log)
        data['total_calls'] = len(data['calls'])

        # Write updated JSON atomically (write to temp file, then rename)
        temp_log_file = log_file.with_suffix('.json.tmp')
        try:
            with open(temp_log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                import os
                os.fsync(f.fileno())  # Force write to disk
            # Atomic rename (on most systems)
            temp_log_file.replace(log_file)
        except Exception as e:
            # Clean up temp file if write failed
            if temp_log_file.exists():
                temp_log_file.unlink()
            raise e

        # Append to markdown file
        with open(md_file, 'a', encoding='utf-8') as f:
            if call_number == 1:
                # Write header for first call
                f.write(f"# LLM Call Logs for Test Case: {test_id}\n\n")
                f.write(f"**Test Started**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")

            f.write(f"## Call #{call_number}: {log_entry['call_type']}\n\n")
            f.write(f"**Model**: {log_entry['model']}\n\n")
            f.write(f"**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log_entry['timestamp']))}\n\n")

            if log_entry['metadata']:
                f.write(f"**Metadata**:\n")
                for key, value in log_entry['metadata'].items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")

            f.write(f"**Messages**:\n")
            for msg in log_entry['messages']:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                f.write(f"\n### {role.upper()}\n")
                f.write(f"```\n{content}\n```\n")

            f.write(f"\n**Response**:\n")
            f.write(f"```\n{log_entry['response']}\n```\n")
            f.write("\n---\n\n")
            f.flush()  # Force write to disk

    except Exception as e:
        # Don't fail the test if logging fails, just print error
        import traceback
        error_msg = f"Warning: Failed to write instant log for {test_id} (call #{call_number}): {e}"
        print(error_msg)
        if not isinstance(e, (json.JSONDecodeError, ValueError)):
            # Print traceback for unexpected errors (but not for JSON decode errors)
            traceback.print_exc()


class LoggingChatLLM(ChatLLM):
    """Wrapper around ChatLLM that logs all calls"""

    def __init__(self, *args, llm_type: str = "default", **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_type = llm_type

    def create_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict]):
        """Override to log tool calls"""
        result = super().create_with_tools(messages, tools)

        # Log the call
        response_content = result.content if hasattr(result, 'content') else str(result)
        tool_calls = result.tool_calls if hasattr(result, 'tool_calls') else None

        # Convert tool_calls to serializable format
        tool_calls_list = []
        if tool_calls:
            for tc in tool_calls:
                try:
                    tool_calls_list.append({
                        'id': tc.id if hasattr(tc, 'id') else None,
                        'type': tc.type if hasattr(tc, 'type') else None,
                        'function': {
                            'name': tc.function.name if hasattr(tc.function, 'name') else str(tc.function),
                            'arguments': tc.function.arguments if hasattr(tc.function, 'arguments') else None
                        } if hasattr(tc, 'function') else None
                    })
                except Exception:
                    # Fallback to string representation
                    tool_calls_list.append(str(tc))

        log_llm_call(
            call_type=f"{self.llm_type}_with_tools",
            model=self.model,
            messages=messages,
            response=response_content,
            metadata={
                'tools': [t.get('function', {}).get('name', 'unknown') for t in tools] if tools else [],
                'tool_calls': tool_calls_list,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        )

        return result

    def chat_to_llm(self, messages: List[Dict[str, Any]]):
        """Override to log regular chat calls"""
        result = super().chat_to_llm(messages)

        # Log the call
        log_llm_call(
            call_type=f"{self.llm_type}_chat",
            model=self.model,
            messages=messages,
            response=result,
            metadata={
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        )

        return result


@contextmanager
def suppress_output():
    """Suppress stdout and stderr completely"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@dataclass
class SimpleConfig:
    benchmark_path: str
    output_dir: str = "./gaia_validation_results"
    max_workers: int = 2
    silent: bool = False


class SimpleGAIAEvaluator:
    """Simple GAIA Benchmark Evaluator"""

    def __init__(self, config: SimpleConfig):
        self.config = config
        self.benchmark_path = Path(config.benchmark_path)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        self.dataset = self._load_dataset()
        self.results = []

        # Create wrapped LLM instances for logging
        self.logged_llm_plan = self._wrap_llm(llm_for_plan, "plan")
        self.logged_llm_act = self._wrap_llm(llm_for_act, "act")
        self.logged_llm_tool = self._wrap_llm(llm_for_tool, "tool")
        self.logged_llm_vision = self._wrap_llm(llm_for_vision, "vision")

    def _wrap_llm(self, original_llm: ChatLLM, llm_type: str) -> LoggingChatLLM:
        """Wrap an LLM instance with logging"""
        wrapped = LoggingChatLLM(
            base_url=original_llm.base_url,
            api_key=original_llm.api_key,
            model=original_llm.model,
            client=original_llm.client,
            max_tokens=original_llm.max_tokens,
            temperature=original_llm.temperature,
            stream=original_llm.stream,
            tools=original_llm.tools,
            llm_type=llm_type
        )
        return wrapped
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load GAIA dataset - prioritizes validation.json"""
        json_files = list(self.benchmark_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.benchmark_path}")

        # Priority order: validation.json > test.json > any JSON with test data
        test_file = None

        # First priority: validation.json
        for json_file in json_files:
            if json_file.name == "validation.json":
                test_file = json_file
                if not self.config.silent:
                    print(f"Using validation dataset: {json_file.name}")
                break

        # Second priority: test.json
        if not test_file:
            for json_file in json_files:
                if json_file.name == "test.json":
                    test_file = json_file
                    if not self.config.silent:
                        print(f"Using test dataset: {json_file.name}")
                    break

        # Third priority: any file with test data structure
        if not test_file:
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Check if this looks like test data (has list of items with 'Question' field)
                    if isinstance(data, list) and data and 'Question' in data[0]:
                        test_file = json_file
                        if not self.config.silent:
                            print(f"Using dataset: {json_file.name}")
                        break
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        # Fallback: first JSON file
        if not test_file:
            test_file = json_files[0]
            if not self.config.silent:
                print(f"Using fallback dataset: {test_file.name}")

        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ensure we have a list
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            else:
                # If it's a dict but no 'data' key, convert to list
                data = [data]

        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")

        if not self.config.silent:
            print(f"Loaded {len(data)} test cases from {test_file.name}")
        return data
    
    def _create_cosight(self, test_id: str) -> CoSight:
        """Create CoSight instance with logged LLM wrappers"""
        workspace = Path(f"./gaia_workspaces/test_{test_id}")
        workspace.mkdir(parents=True, exist_ok=True)

        return CoSight(
            plan_llm=self.logged_llm_plan,
            act_llm=self.logged_llm_act,
            tool_llm=self.logged_llm_tool,
            vision_llm=self.logged_llm_vision,
            work_space_path=str(workspace),
            message_uuid=f"test_{test_id}"
        )
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer from response"""
        if isinstance(response, dict):
            for key in ['answer', 'final_answer', 'result']:
                if key in response:
                    return str(response[key])
            return str(response)
        
        response_str = str(response)
        
        # Simple pattern matching
        import re
        patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'The answer is:\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_str, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Return last sentence
        sentences = response_str.split('.')
        return sentences[-1].strip() if sentences else response_str.strip()
    
    def _is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Check if answer is correct"""
        if not predicted or not ground_truth:
            return False
        
        pred_norm = predicted.lower().strip()
        gt_norm = ground_truth.lower().strip()
        
        # Exact match or contains
        return pred_norm == gt_norm or gt_norm in pred_norm or pred_norm in gt_norm
    
    def evaluate_case(self, test_case: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Evaluate single test case"""
        global _llm_call_logs, _current_test_id, _current_output_dir

        test_id = test_case.get('id', f"case_{index}")
        question = test_case.get('Question', test_case.get('question', ''))
        ground_truth = test_case.get('Final answer', test_case.get('ground_truth', ''))

        # Set up instant logging for this test case
        _llm_call_logs = []
        _current_test_id = test_id
        _current_output_dir = str(self.output_dir)

        start_time = time.time()

        try:
            cosight = self._create_cosight(test_id)
            response = cosight.execute(question)
            predicted = self._extract_answer(response)
            execution_time = time.time() - start_time
            is_correct = self._is_correct(predicted, ground_truth)

            if not self.config.silent:
                print(f"Case {index + 1}: {'✓' if is_correct else '✗'} ({execution_time:.1f}s) - {len(_llm_call_logs)} LLM calls")

            return {
                'test_id': test_id,
                'index': index,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'is_correct': is_correct,
                'execution_time': execution_time,
                'llm_call_count': len(_llm_call_logs),
                'error': None
            }

        except Exception as e:
            execution_time = time.time() - start_time
            if not self.config.silent:
                print(f"Case {index + 1}: Error - {str(e)[:50]}... - {len(_llm_call_logs)} LLM calls")

            return {
                'test_id': test_id,
                'index': index,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': '',
                'is_correct': False,
                'execution_time': execution_time,
                'llm_call_count': len(_llm_call_logs),
                'error': str(e)
            }
        finally:
            # Clean up global state
            _current_test_id = None
            _current_output_dir = None
    
    def run_evaluation(self, max_cases: Optional[int] = None) -> Dict[str, Any]:
        """Run evaluation"""
        try:
            if max_cases is not None:
                test_cases = self.dataset[:max_cases]
            else:
                test_cases = self.dataset
            
            if not self.config.silent:
                print(f"Evaluating {len(test_cases)} test cases...")
        except Exception as e:
            print(f"Error in run_evaluation: {e}")
            print(f"max_cases: {max_cases}, type: {type(max_cases)}")
            print(f"dataset type: {type(self.dataset)}")
            raise
        
        # Simple sequential evaluation (no parallel to avoid complexity)
        for i, test_case in enumerate(test_cases):
            result = self.evaluate_case(test_case, i)
            self.results.append(result)
        
        # Calculate metrics
        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])
        errors = sum(1 for r in self.results if r['error'])
        avg_time = sum(r['execution_time'] for r in self.results) / total if total > 0 else 0
        total_llm_calls = sum(r.get('llm_call_count', 0) for r in self.results)
        avg_llm_calls = total_llm_calls / total if total > 0 else 0

        metrics = {
            'total_cases': total,
            'correct_cases': correct,
            'accuracy': correct / total if total > 0 else 0,
            'error_cases': errors,
            'avg_execution_time': avg_time,
            'total_llm_calls': total_llm_calls,
            'avg_llm_calls_per_case': avg_llm_calls
        }
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _save_results(self, metrics: Dict[str, Any]):
        """Save results"""
        timestamp = int(time.time())
        
        # Save JSON
        results_file = self.output_dir / f"gaia_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'metrics': metrics,
                'timestamp': timestamp
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎯 Results saved to {results_file}")
        print(f"📊 Accuracy: {metrics['accuracy']:.1%} ({metrics['correct_cases']}/{metrics['total_cases']})")
        print(f"⏱️  Avg time: {metrics['avg_execution_time']:.1f}s")
        print(f"🤖 Total LLM calls: {metrics['total_llm_calls']} (avg: {metrics['avg_llm_calls_per_case']:.1f} per case)")
        print(f"❌ Errors: {metrics['error_cases']}")
        print(f"📝 LLM logs saved to {self.output_dir}/llm_logs_*.json and *.md")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simple GAIA benchmark evaluation')
    parser.add_argument('--benchmark_path', required=True, help='Path to GAIA dataset')
    parser.add_argument('--output_dir', default='./gaia_validation_results', help='Output directory (default: gaia_validation_results)')
    parser.add_argument('--max_cases', type=int, help='Max test cases to evaluate')
    parser.add_argument('--silent', action='store_true', help='Silent mode - minimal output')
    
    args = parser.parse_args()
    
    if not Path(args.benchmark_path).exists():
        print(f"Error: Path does not exist: {args.benchmark_path}")
        return 1
    
    try:
        config = SimpleConfig(
            benchmark_path=args.benchmark_path,
            output_dir=args.output_dir,
            silent=args.silent
        )
        
        evaluator = SimpleGAIAEvaluator(config)
        metrics = evaluator.run_evaluation(max_cases=args.max_cases)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())