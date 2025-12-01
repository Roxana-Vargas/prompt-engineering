"""
Example: Mathematical reasoning with different prompt techniques
Demonstrates Chain-of-Thought vs Zero-Shot prompting
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_templates.techniques import get_technique
from src.utils.llm_client import get_client
from src.utils.evaluator import PromptEvaluator


def math_scorer(response: str) -> float:
    """Simple scorer for math problems - checks if answer contains a number"""
    import re
    numbers = re.findall(r'\d+', response)
    if numbers:
        return 1.0
    return 0.0


def main():
    # Initialize client
    client = get_client("openai", model="gpt-4")
    evaluator = PromptEvaluator(client)
    
    # Mathematical reasoning task
    task = "If a train travels 300 miles in 4 hours, and then travels another 200 miles in 3 hours, what is the average speed for the entire journey?"
    
    # Build prompts with different techniques
    zero_shot = get_technique("zero_shot")
    cot = get_technique("chain_of_thought")
    
    techniques = {
        "Zero-Shot": zero_shot.build_prompt(task),
        "Chain-of-Thought": cot.build_prompt(
            task,
            instruction="Solve this math problem step by step, showing all your work."
        )
    }
    
    # Compare techniques
    print("Evaluating mathematical reasoning task...")
    comparison = evaluator.compare_techniques(
        task=task,
        techniques=techniques,
        scorer=math_scorer,
        temperature=0.7,
        max_tokens=500
    )
    
    # Generate and print report
    report = evaluator.generate_report(comparison)
    print(report)
    
    return comparison


if __name__ == "__main__":
    main()

