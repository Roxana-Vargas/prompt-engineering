"""
Comprehensive comparison of multiple prompt techniques
Demonstrates the full evaluation system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_templates.techniques import get_technique
from src.utils.llm_client import get_client
from src.utils.evaluator import PromptEvaluator
from src.utils.config import Config


def logical_reasoning_scorer(response: str) -> float:
    """Score based on logical reasoning quality"""
    score = 0.0
    
    # Check for step-by-step reasoning
    if "step" in response.lower() or "first" in response.lower():
        score += 0.3
    
    # Check for conclusion
    if "therefore" in response.lower() or "conclusion" in response.lower() or "answer" in response.lower():
        score += 0.3
    
    # Check for logical connectors
    logical_words = ["because", "since", "if", "then", "therefore", "thus"]
    if any(word in response.lower() for word in logical_words):
        score += 0.4
    
    return min(score, 1.0)


def main():
    # Validate configuration
    Config.validate()
    
    # Initialize client
    client = get_client("openai", model="gpt-4")
    evaluator = PromptEvaluator(client)
    
    # Complex logical reasoning task
    task = """
    All roses are flowers. Some flowers fade quickly. Therefore, some roses fade quickly.
    Is this reasoning valid? Explain why or why not.
    """
    
    # CoT examples with structured step-by-step reasoning
    cot_examples = [
        {
            "input": "All birds can fly. Penguins are birds. Can penguins fly?",
            "reasoning": "Step 1: Identify the premises\n- Premise 1: All birds can fly\n- Premise 2: Penguins are birds\n\nStep 2: Apply logical reasoning\n- If all birds can fly (Premise 1)\n- And penguins are birds (Premise 2)\n- Then logically: penguins can fly\n\nStep 3: Check against real-world knowledge\n- We know from facts that penguins cannot fly\n- This creates a contradiction\n\nStep 4: Analyze the reasoning\n- The logical structure is valid (if-then reasoning)\n- But the first premise is factually incorrect\n\nStep 5: Conclusion\n- The reasoning is logically valid but factually incorrect",
            "output": "The reasoning is logically valid (if all birds can fly and penguins are birds, then penguins can fly), but it's factually incorrect because penguins cannot fly."
        }
    ]
    
    # Build prompts for all techniques
    techniques = {
        "Zero-Shot": get_technique("zero_shot").build_prompt(
            task,
            instruction="Analyze the logical reasoning and determine if it's valid."
        ),
        "Chain-of-Thought": get_technique("chain_of_thought").build_prompt(
            task,
            instruction="Analyze this logical reasoning step by step.",
            examples=cot_examples
        ),
        "Self-Consistency": get_technique("self_consistency").build_prompt(
            task,
            instruction="Analyze this logical reasoning. Consider multiple perspectives."
        ),
        "Tree of Thoughts": get_technique("tree_of_thoughts").build_prompt(
            task,
            instruction="Analyze this logical reasoning. Consider multiple perspectives."
        )       
    }
    
    # Compare all techniques
    print("=" * 80)
    print("COMPREHENSIVE PROMPT TECHNIQUE COMPARISON")
    print("=" * 80)
    print("\nTask:", task.strip())
    print("\nEvaluating multiple techniques...\n")
    
    comparison = evaluator.compare_techniques(
        task=task,
        techniques=techniques,
        scorer=logical_reasoning_scorer,
        temperature=0.7,
        max_tokens=600
    )
    
    # Generate detailed report
    report = evaluator.generate_report(comparison)
    print(report)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for result in comparison.results:
        print(f"\n{result.technique}:")
        print(f"  Time: {result.execution_time:.2f}s")
        print(f"  Tokens: {result.token_count}")
        print(f"  Score: {result.score:.2f}" if result.score else "  Score: N/A")
    
    if comparison.best_technique:
        print(f"\n🏆 Best Technique: {comparison.best_technique}")
    
    return comparison


if __name__ == "__main__":
    main()

