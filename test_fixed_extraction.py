#!/usr/bin/env python3
"""Test the FIXED extraction method"""

import re

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

def extract_answer_fixed(response: str) -> str:
    """FIXED extraction method"""
    response_str = str(response)

    # First, try to extract from CoSight's Summary section (capture everything after Summary:)
    summary_match = re.search(r'Summary:\s*\n(.+)', response_str, re.IGNORECASE | re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
        # Try to find explicit answer patterns within the summary
        answer_patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',  # Most specific pattern first
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

print("=" * 60)
print("Testing FIXED Extraction Method")
print("=" * 60)
result = extract_answer_fixed(test_response)
print(f"Extracted: '{result}'")
print(f"Expected: 'egalitarian'")
print(f"Match: {result == 'egalitarian'}")
