#!/usr/bin/env python3
"""
GAIA Benchmark Evaluation Script for CoSight

This script integrates CoSight with the GAIA benchmark for comprehensive evaluation.
GAIA (General AI Assistant) is a benchmark for evaluating AI assistants on real-world tasks.

Usage:
    python gaia_evaluation.py --benchmark_path /path/to/gaia --output_dir ./results
"""

import os
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from CoSight import CoSight
from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision
from app.common.logger_util import logger


class GAIAEvaluator:
    """GAIA Benchmark Evaluator for CoSight"""
    
    def __init__(self, benchmark_path: str, output_dir: str, workspace_base: str = None):
        """
        Initialize GAIA Evaluator
        
        Args:
            benchmark_path: Path to GAIA benchmark dataset
            output_dir: Directory to save evaluation results
            workspace_base: Base directory for CoSight workspaces
        """
        self.benchmark_path = Path(benchmark_path)
        self.output_dir = Path(output_dir)
        self.workspace_base = Path(workspace_base) if workspace_base else Path("./gaia_workspaces")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        
        # Load GAIA dataset
        self.dataset = self._load_dataset()
        
        # Results storage
        self.results = []
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load GAIA benchmark dataset"""
        try:
            # Look for GAIA dataset files
            possible_files = [
                self.benchmark_path / "test.json",
                self.benchmark_path / "test.jsonl", 
                self.benchmark_path / "gaia_test.json",
                self.benchmark_path / "gaia_test.jsonl"
            ]
            
            dataset_file = None
            for file_path in possible_files:
                if file_path.exists():
                    dataset_file = file_path
                    break
            
            if not dataset_file:
                raise FileNotFoundError(f"GAIA dataset not found in {self.benchmark_path}")
            
            logger.info(f"Loading GAIA dataset from {dataset_file}")
            
            # Load dataset based on file extension
            if dataset_file.suffix == '.json':
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif dataset_file.suffix == '.jsonl':
                data = []
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                raise ValueError(f"Unsupported file format: {dataset_file.suffix}")
            
            logger.info(f"Loaded {len(data)} test cases from GAIA dataset")
            return data
            
        except Exception as e:
            logger.error(f"Error loading GAIA dataset: {e}")
            raise
    
    def _create_cosight_instance(self, test_case_id: str) -> CoSight:
        """Create a new CoSight instance for a test case"""
        # Create unique workspace for this test case
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        workspace_path = self.workspace_base / f"gaia_test_{test_case_id}_{timestamp}"
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create CoSight instance
        cosight = CoSight(
            plan_llm=llm_for_plan,
            act_llm=llm_for_act,
            tool_llm=llm_for_tool,
            vision_llm=llm_for_vision,
            work_space_path=str(workspace_path),
            message_uuid=f"gaia_test_{test_case_id}"
        )
        
        return cosight
    
    def _extract_answer_from_response(self, response: str, test_case: Dict[str, Any]) -> str:
        """Extract the final answer from CoSight response"""
        # This is a simple extraction - you may need to customize based on GAIA requirements
        # GAIA typically expects specific answer formats
        
        # Try to find structured answer in response
        if isinstance(response, dict):
            # If response is a dict, look for common answer keys
            answer_keys = ['answer', 'final_answer', 'result', 'conclusion', 'summary']
            for key in answer_keys:
                if key in response:
                    return str(response[key])
            return str(response)
        
        # If response is a string, try to extract the final answer
        response_str = str(response)
        
        # Look for common answer patterns
        answer_patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'Result:\s*(.+?)(?:\n|$)',
            r'Conclusion:\s*(.+?)(?:\n|$)',
        ]
        
        import re
        for pattern in answer_patterns:
            match = re.search(pattern, response_str, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return the last few sentences
        sentences = response_str.split('.')
        if len(sentences) > 3:
            return '. '.join(sentences[-3:]).strip()
        
        return response_str.strip()
    
    def evaluate_single_case(self, test_case: Dict[str, Any], case_index: int) -> Dict[str, Any]:
        """Evaluate a single GAIA test case"""
        test_case_id = test_case.get('id', f"case_{case_index}")
        question = test_case.get('Question', test_case.get('question', ''))
        ground_truth = test_case.get('Final answer', test_case.get('ground_truth', ''))
        
        logger.info(f"Evaluating test case {case_index + 1}/{len(self.dataset)}: {test_case_id}")
        
        start_time = time.time()
        
        try:
            # Create CoSight instance
            cosight = self._create_cosight_instance(test_case_id)
            
            # Execute the question
            response = cosight.execute(question)
            
            # Extract answer
            predicted_answer = self._extract_answer_from_response(response, test_case)
            
            execution_time = time.time() - start_time
            
            # Calculate metrics (simplified - you may need more sophisticated evaluation)
            is_correct = self._evaluate_correctness(predicted_answer, ground_truth)
            
            result = {
                'test_case_id': test_case_id,
                'case_index': case_index,
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'response': response,
                'is_correct': is_correct,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Test case {test_case_id} completed in {execution_time:.2f}s - Correct: {is_correct}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating test case {test_case_id}: {e}")
            return {
                'test_case_id': test_case_id,
                'case_index': case_index,
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': '',
                'response': '',
                'is_correct': False,
                'execution_time': time.time() - start_time,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _evaluate_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Evaluate if the predicted answer is correct"""
        # This is a simplified evaluation - GAIA typically requires more sophisticated evaluation
        # You may need to implement specific evaluation logic based on GAIA requirements
        
        if not predicted or not ground_truth:
            return False
        
        # Normalize strings for comparison
        predicted_norm = predicted.lower().strip()
        ground_truth_norm = ground_truth.lower().strip()
        
        # Exact match
        if predicted_norm == ground_truth_norm:
            return True
        
        # Check if ground truth is contained in predicted answer
        if ground_truth_norm in predicted_norm:
            return True
        
        # Check if predicted answer is contained in ground truth
        if predicted_norm in ground_truth_norm:
            return True
        
        # For numerical answers, try to extract and compare numbers
        try:
            import re
            pred_numbers = re.findall(r'-?\d+\.?\d*', predicted_norm)
            gt_numbers = re.findall(r'-?\d+\.?\d*', ground_truth_norm)
            
            if pred_numbers and gt_numbers:
                # Compare the first number found
                pred_num = float(pred_numbers[0])
                gt_num = float(gt_numbers[0])
                return abs(pred_num - gt_num) < 0.01  # Allow small numerical differences
        except:
            pass
        
        return False
    
    def run_evaluation(self, max_cases: Optional[int] = None, start_index: int = 0) -> Dict[str, Any]:
        """Run the complete GAIA evaluation"""
        logger.info(f"Starting GAIA evaluation with {len(self.dataset)} test cases")
        
        # Limit number of test cases if specified
        test_cases = self.dataset[start_index:]
        if max_cases:
            test_cases = test_cases[:max_cases]
        
        logger.info(f"Evaluating {len(test_cases)} test cases (starting from index {start_index})")
        
        # Evaluate each test case
        for i, test_case in enumerate(test_cases):
            result = self.evaluate_single_case(test_case, start_index + i)
            self.results.append(result)
            
            # Save intermediate results every 10 cases
            if (i + 1) % 10 == 0:
                self._save_intermediate_results()
        
        # Calculate final metrics
        metrics = self._calculate_metrics()
        
        # Save final results
        self._save_final_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        if not self.results:
            return {}
        
        total_cases = len(self.results)
        correct_cases = sum(1 for r in self.results if r.get('is_correct', False))
        accuracy = correct_cases / total_cases if total_cases > 0 else 0
        
        execution_times = [r.get('execution_time', 0) for r in self.results]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        error_cases = sum(1 for r in self.results if 'error' in r)
        
        metrics = {
            'total_cases': total_cases,
            'correct_cases': correct_cases,
            'accuracy': accuracy,
            'error_cases': error_cases,
            'avg_execution_time': avg_execution_time,
            'total_execution_time': sum(execution_times),
            'min_execution_time': min(execution_times) if execution_times else 0,
            'max_execution_time': max(execution_times) if execution_times else 0
        }
        
        return metrics
    
    def _save_intermediate_results(self):
        """Save intermediate results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        intermediate_file = self.output_dir / f"gaia_results_intermediate_{timestamp}.json"
        
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'metrics': self._calculate_metrics(),
                'timestamp': timestamp
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Intermediate results saved to {intermediate_file}")
    
    def _save_final_results(self, metrics: Dict[str, Any]):
        """Save final evaluation results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = self.output_dir / f"gaia_results_detailed_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'metrics': metrics,
                'timestamp': timestamp,
                'benchmark_path': str(self.benchmark_path)
            }, f, indent=2, ensure_ascii=False)
        
        # Save summary results
        summary_file = self.output_dir / f"gaia_results_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'timestamp': timestamp,
                'benchmark_path': str(self.benchmark_path)
            }, f, indent=2, ensure_ascii=False)
        
        # Save CSV for easy analysis
        csv_file = self.output_dir / f"gaia_results_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Final results saved:")
        logger.info(f"  - Detailed: {results_file}")
        logger.info(f"  - Summary: {summary_file}")
        logger.info(f"  - CSV: {csv_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("GAIA EVALUATION SUMMARY")
        print("="*50)
        print(f"Total test cases: {metrics['total_cases']}")
        print(f"Correct answers: {metrics['correct_cases']}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Error cases: {metrics['error_cases']}")
        print(f"Average execution time: {metrics['avg_execution_time']:.2f}s")
        print(f"Total execution time: {metrics['total_execution_time']:.2f}s")
        print("="*50)


def main():
    """Main function for GAIA evaluation"""
    parser = argparse.ArgumentParser(description='Run GAIA benchmark evaluation with CoSight')
    parser.add_argument('--benchmark_path', required=True, help='Path to GAIA benchmark dataset')
    parser.add_argument('--output_dir', default='./gaia_results', help='Output directory for results')
    parser.add_argument('--workspace_base', default='./gaia_workspaces', help='Base directory for CoSight workspaces')
    parser.add_argument('--max_cases', type=int, help='Maximum number of test cases to evaluate')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index for test cases')
    
    args = parser.parse_args()
    
    # Validate benchmark path
    if not os.path.exists(args.benchmark_path):
        print(f"Error: Benchmark path does not exist: {args.benchmark_path}")
        return 1
    
    try:
        # Create evaluator
        evaluator = GAIAEvaluator(
            benchmark_path=args.benchmark_path,
            output_dir=args.output_dir,
            workspace_base=args.workspace_base
        )
        
        # Run evaluation
        metrics = evaluator.run_evaluation(
            max_cases=args.max_cases,
            start_index=args.start_index
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
