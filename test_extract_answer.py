#!/usr/bin/env python3
"""Test the _extract_answer method with CoSight response format"""

import re


def extract_answer(response: str) -> str:
    """Extract answer from response"""
    if isinstance(response, dict):
        for key in ['answer', 'final_answer', 'result']:
            if key in response:
                return str(response[key])
        return str(response)

    response_str = str(response)

    # Simple pattern matching
    import re

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

    # Fallback to original patterns for other response formats
    patterns = [
        r'Final Answer:\s*(.+?)(?:\n|$)',
        r'Answer:\s*(.+?)(?:\n|$)',
        r'The answer is:\s*(.+?)(?:\n|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_str, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # Return last non-empty line as fallback
    lines = [line.strip() for line in response_str.split('\n') if line.strip()]
    return lines[-1] if lines else response_str.strip()


# Test with CoSight format
test_response_1 = """
Task:
A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?

Plan Status:
Step 1: Search for AI regulation papers (completed)
Step 2: Analyze results (completed)

Summary:
The answer is: egalitarian
"""

test_response_2 = """
Task:
What is 2 + 2?

Plan Status:
Step 1: Calculate (completed)

Summary:
Based on the calculation, the result is 4.
"""

test_response_3 = """
Task:
What is the capital of France?

Plan Status:
Step 1: Search (completed)

Summary:
Paris
"""

print("Test 1 - With 'The answer is:' pattern:")
result1 = extract_answer(test_response_1)
print(f"Extracted: '{result1}'")
print(f"Expected: 'egalitarian'")
print(f"Match: {result1 == 'egalitarian'}")
print()

print("Test 2 - With 'result is' pattern:")
result2 = extract_answer(test_response_2)
print(f"Extracted: '{result2}'")
print(f"Expected: '4' (or similar)")
print()

print("Test 3 - Just the answer:")
result3 = extract_answer(test_response_3)
print(f"Extracted: '{result3}'")
print(f"Expected: 'Paris'")
print(f"Match: {result3 == 'Paris'}")
