"""Data source. Modify this file for different tasks."""

# Example data - replace with your dataset
PROMPTS = [
    "What is 2 + 2?",
    "What is 5 * 3?",
    "What is 10 - 4?",
    "What is 12 / 3?",
    "What is 7 + 8?",
    "What is 9 * 2?",
    "What is 20 - 11?",
    "What is 15 / 5?",
]

ANSWERS = ["4", "15", "6", "4", "15", "18", "9", "3"]

current_idx = 0


def get_batch(size=8):
    """Get next batch of prompts."""
    global current_idx

    indices = [(current_idx + i) % len(PROMPTS) for i in range(size)]
    current_idx = (current_idx + size) % len(PROMPTS)

    return {
        'prompts': [PROMPTS[i] for i in indices],
        'ground_truths': [ANSWERS[i] for i in indices],
    }


# For multi-turn, return prompts as message lists:
# def get_batch(size=8):
#     return {
#         'prompts': [
#             [
#                 {'role': 'user', 'content': 'What is 2+2?'},
#                 {'role': 'assistant', 'content': 'The answer is 4.'},
#                 {'role': 'user', 'content': 'Now multiply that by 3.'},
#             ]
#         ] * size,
#         'ground_truths': ['12'] * size,
#     }


# For tool calling, prompts that require tool use:
# TOOL_PROMPTS = [
#     "What is 15 + 27? Use the calculator tool.",
#     "What is 100 divided by 4? Use the calculator.",
#     "Search for information about Python programming.",
#     "What's the weather in Tokyo?",
# ]
# TOOL_ANSWERS = ["42", "25", None, None]  # None = no exact answer, reward based on tool usage
#
# def get_batch(size=4):
#     return {
#         'prompts': TOOL_PROMPTS[:size],
#         'ground_truths': TOOL_ANSWERS[:size],
#     }
#
# def compute_reward(responses, ground_truths):
#     """Reward for correct tool usage."""
#     import re
#     rewards = []
#     for resp, gt in zip(responses, ground_truths):
#         # Check if model called a tool
#         has_tool_call = bool(re.search(r'<tool_call>|<function>', resp, re.IGNORECASE))
#         if gt:
#             # If we have ground truth, check answer
#             rewards.append(1.0 if str(gt) in resp else 0.0)
#         else:
#             # No ground truth, reward tool usage
#             rewards.append(1.0 if has_tool_call else 0.0)
#     return rewards
