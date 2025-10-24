#!/usr/bin/env python3
"""Test script to see what CoSight response format looks like"""

# Sample response based on the finalize_plan format
test_response = """
Task:
A paper about AI regulation that was originally submitted to arXiv.org in June 2022

Plan Status:
Step 1: Research (completed)
Step 2: Analysis (completed)
...
Step 12: Final answer (completed)

Summary:
Task Summary Report

After thorough research and analysis, I found the answer to your question.

The word "egalitarian" appears in both:
1. The AI regulation paper from June 2022 (as an axis label)
2. The Physics and Society article from August 11, 2016 (describing a type of society)

Final Answer: egalitarian
"""

import re

def extract_answer_current(response: str) -> str:
    """Current extraction method"""
    response_str = str(response)

    # First, try to extract from CoSight's Summary section
    summary_match = re.search(r'Summary:\s*\n(.+?)(?:\n\n|\Z)', response_str, re.IGNORECASE | re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
        # Try to find explicit answer patterns within the summary
        answer_patterns = [
            r'(?:final answer|answer|the answer is)[:\s]+(.+?)(?:\n|$)',
            r'(?:result|conclusion)[:\s]+(.+?)(?:\n|$)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, summary, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        # If no pattern matches, return the entire summary
        return summary

    return response_str.strip()


def extract_answer_improved(response: str) -> str:
    """Improved extraction method"""
    response_str = str(response)

    # First, try to extract from CoSight's Summary section
    summary_match = re.search(r'Summary:\s*\n(.+?)(?:\n\n|\Z)', response_str, re.IGNORECASE | re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()

        # Try to find explicit answer patterns within the summary
        answer_patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',  # More specific pattern
            r'(?:final answer|answer)[:\s]+(.+?)(?:\n|$)',
            r'(?:result|conclusion)[:\s]+(.+?)(?:\n|$)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, summary, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no pattern matches, try to get the last meaningful line
        lines = [line.strip() for line in summary.split('\n') if line.strip()]
        # Skip lines that are section headers or generic statements
        skip_keywords = ['task summary', 'report', 'after', 'analysis', 'research', 'found']
        for line in reversed(lines):
            if not any(keyword in line.lower() for keyword in skip_keywords):
                return line

        # If all else fails, return the entire summary
        return summary

    return response_str.strip()


print("=" * 60)
print("Testing Current Extraction Method")
print("=" * 60)
result = extract_answer_current(test_response)
print(f"Extracted: '{result}'")
print()

print("=" * 60)
print("Testing Improved Extraction Method")
print("=" * 60)
result = extract_answer_improved(test_response)
print(f"Extracted: '{result}'")
print()

print("=" * 60)
print("Summary section only:")
print("=" * 60)
summary_match = re.search(r'Summary:\s*\n(.+?)(?:\n\n|\Z)', test_response, re.IGNORECASE | re.DOTALL)
if summary_match:
    print(summary_match.group(1))
