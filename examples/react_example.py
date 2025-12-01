"""
Example: ReAct (Reasoning + Acting) prompting
Demonstrates iterative reasoning and action-taking
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_templates.techniques import get_technique
from src.utils.llm_client import get_client
from src.utils.evaluator import PromptEvaluator


def main():
    # Initialize client
    client = get_client("openai", model="gpt-4")
    evaluator = PromptEvaluator(client)
    
    # Problem-solving task that benefits from reasoning and actions
    task = "I need to plan a trip from New York to Los Angeles. What's the best way to get there considering time and cost?"
    
    # Available actions/tools
    available_actions = [
        "search_flights(departure, destination, date)",
        "search_trains(departure, destination, date)",
        "search_buses(departure, destination, date)",
        "calculate_distance(origin, destination)",
        "get_weather(location, date)"
    ]
    
    # ReAct examples showing the iterative loop
    react_examples = [
        {
            "problem": "Find the best way to travel from Boston to Washington DC",
            "thought": "I need to find the best transportation option. Let me think about what information I need: distance, time, and cost.",
            "action": "calculate_distance('Boston', 'Washington DC')",
            "observation": "Distance: ~440 miles",
            "thought2": "The distance is moderate. I should compare multiple options: flights, trains, and driving. Let me check flight options first.",
            "action2": "search_flights('Boston', 'Washington DC', '2024-03-15')",
            "observation2": "Flights available: $150-$300, 1.5 hours flight time",
            "answer": "Based on the information: Flights are fastest (1.5h) but cost $150-$300. Trains take ~7h but cost $50-$150. Driving takes ~7h and costs ~$60 in gas. For this distance, train or flight would be best depending on budget."
        }
    ]
    
    # Build ReAct prompt
    react = get_technique("react")
    
    techniques = {
        "ReAct": react.build_prompt(
            task=task,
            available_actions=available_actions,
            examples=react_examples
        )
    }
    
    # Evaluate
    print("Evaluating ReAct prompting for problem-solving...")
    comparison = evaluator.compare_techniques(
        task=task,
        techniques=techniques,
        temperature=0.7,
        max_tokens=800
    )
    
    # Generate and print report
    report = evaluator.generate_report(comparison)
    print(report)
    
    return comparison


if __name__ == "__main__":
    main()

