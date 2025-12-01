"""
Example: Text analysis with Few-Shot learning
Demonstrates how few-shot examples improve performance
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_templates.techniques import get_technique
from src.utils.llm_client import get_client
from src.utils.evaluator import PromptEvaluator


def sentiment_scorer(response: str) -> float:
    """Simple sentiment scorer"""
    positive_words = ["positive", "good", "great", "excellent", "happy"]
    negative_words = ["negative", "bad", "terrible", "awful", "sad"]
    
    response_lower = response.lower()
    positive_count = sum(1 for word in positive_words if word in response_lower)
    negative_count = sum(1 for word in negative_words if word in response_lower)
    
    if positive_count > negative_count:
        return 1.0
    elif negative_count > positive_count:
        return 0.5
    return 0.0


def main():
    # Initialize client
    client = get_client("openai", model="gpt-4")
    evaluator = PromptEvaluator(client)
    
    # Sentiment analysis task
    task = "Analyze the sentiment of this text: 'I just got promoted and I'm so excited about the new opportunities!'"
    
    # Few-shot examples with reasoning
    examples = [
        {
            "input": "I hate this product, it's terrible.",
            "reasoning": "The words 'hate' and 'terrible' are strong negative indicators. The overall tone is clearly negative.",
            "output": "Negative sentiment. The text expresses dissatisfaction and disappointment."
        },
        {
            "input": "This is the best day of my life!",
            "reasoning": "The phrase 'best day of my life' with an exclamation mark indicates extreme positive emotion. 'Best' is a superlative positive word.",
            "output": "Positive sentiment. The text expresses extreme happiness and joy."
        },
        {
            "input": "The weather is okay today.",
            "reasoning": "The word 'okay' is neutral - neither strongly positive nor negative. There are no emotional indicators in either direction.",
            "output": "Neutral sentiment. The text expresses neither strong positive nor negative feelings."
        }
    ]
    
    # Build prompts
    zero_shot = get_technique("zero_shot")
    few_shot = get_technique("few_shot")
    
    techniques = {
        "Zero-Shot": zero_shot.build_prompt(
            task,
            instruction="Analyze the sentiment of the given text and classify it as positive, negative, or neutral."
        ),
        "Few-Shot": few_shot.build_prompt(
            task,
            examples=examples,
            instruction="Analyze the sentiment of the given text and classify it as positive, negative, or neutral."
        )
    }
    
    # Compare techniques
    print("Evaluating sentiment analysis task...")
    comparison = evaluator.compare_techniques(
        task=task,
        techniques=techniques,
        scorer=sentiment_scorer,
        temperature=0.3,
        max_tokens=200
    )
    
    # Generate and print report
    report = evaluator.generate_report(comparison)
    print(report)
    
    return comparison


if __name__ == "__main__":
    main()

