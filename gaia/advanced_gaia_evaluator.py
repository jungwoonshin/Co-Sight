#!/usr/bin/env python3
"""
Simple GAIA Benchmark Runner for CoSight
"""

import json
import time
import logging
import os
import sys
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
os.environ['PYTHONWARNINGS'] = 'ignore'

from CoSight import CoSight
from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision


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
    output_dir: str = "./gaia_results"
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
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load GAIA dataset"""
        json_files = list(self.benchmark_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.benchmark_path}")
        
        # Look for test.json specifically, or any file that contains test cases
        test_file = None
        for json_file in json_files:
            if json_file.name == "test.json":
                test_file = json_file
                break
        
        # If no test.json found, try to find a file with actual test data
        if not test_file:
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Check if this looks like test data (has list of items with 'Question' field)
                if isinstance(data, list) and data and 'Question' in data[0]:
                    test_file = json_file
                    break
        
        if not test_file:
            # Fallback to first file
            test_file = json_files[0]
        
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
            print(f"Loaded {len(data)} test cases")
        return data
    
    def _create_cosight(self, test_id: str) -> CoSight:
        """Create CoSight instance"""
        workspace = Path(f"./gaia_workspaces/test_{test_id}")
        workspace.mkdir(parents=True, exist_ok=True)
        
        return CoSight(
            plan_llm=llm_for_plan,
            act_llm=llm_for_act,
            tool_llm=llm_for_tool,
            vision_llm=llm_for_vision,
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
        test_id = test_case.get('id', f"case_{index}")
        question = test_case.get('Question', test_case.get('question', ''))
        ground_truth = test_case.get('Final answer', test_case.get('ground_truth', ''))
        
        start_time = time.time()
        
        try:
            cosight = self._create_cosight(test_id)
            response = cosight.execute(question)
            predicted = self._extract_answer(response)
            execution_time = time.time() - start_time
            is_correct = self._is_correct(predicted, ground_truth)
            
            if not self.config.silent:
                print(f"Case {index + 1}: {'✓' if is_correct else '✗'} ({execution_time:.1f}s)")
            
            return {
                'test_id': test_id,
                'index': index,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'is_correct': is_correct,
                'execution_time': execution_time,
                'error': None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            if not self.config.silent:
                print(f"Case {index + 1}: Error - {str(e)[:50]}...")
            
            return {
                'test_id': test_id,
                'index': index,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': '',
                'is_correct': False,
                'execution_time': execution_time,
                'error': str(e)
            }
    
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
        
        metrics = {
            'total_cases': total,
            'correct_cases': correct,
            'accuracy': correct / total if total > 0 else 0,
            'error_cases': errors,
            'avg_execution_time': avg_time
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
        print(f"❌ Errors: {metrics['error_cases']}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simple GAIA benchmark evaluation')
    parser.add_argument('--benchmark_path', required=True, help='Path to GAIA dataset')
    parser.add_argument('--output_dir', default='./gaia_results', help='Output directory')
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