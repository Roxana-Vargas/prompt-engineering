"""
Main entry point for the Prompt Engineering Toolkit
Demonstrates comprehensive usage of all techniques
"""

import argparse
from src.prompt_templates.techniques import get_technique, TECHNIQUES
from src.utils.llm_client import get_client
from src.utils.evaluator import PromptEvaluator
from src.utils.config import Config


def run_example(example_name: str):
    """Run a specific example"""
    if example_name == "math":
        from examples.math_reasoning import main
        main()
    elif example_name == "text":
        from examples.text_analysis import main
        main()
    elif example_name == "react":
        from examples.react_example import main
        main()
    elif example_name == "comprehensive":
        from examples.comprehensive_comparison import main
        main()
    else:
        print(f"Unknown example: {example_name}")
        print("Available examples: math, text, react, comprehensive")


def list_techniques():
    """List all available techniques"""
    print("\nAvailable Prompt Engineering Techniques:")
    print("=" * 60)
    for key, technique in TECHNIQUES.items():
        print(f"\n{technique.name} ({key}):")
        print(f"  {technique.description}")


def interactive_mode():
    """Interactive mode for testing prompts"""
    print("\n" + "=" * 60)
    print("Interactive Prompt Engineering Mode")
    print("=" * 60)
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("Please set your API keys in .env file")
        return
    
    # Get client
    provider = input("\nSelect provider (openai/anthropic) [openai]: ").strip() or "openai"
    try:
        client = get_client(provider)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    evaluator = PromptEvaluator(client)
    
    # Get task
    print("\n" + "-" * 60)
    task = input("Enter your task: ").strip()
    if not task:
        print("No task provided. Exiting.")
        return
    
    # Select technique
    print("\nAvailable techniques:")
    for i, (key, technique) in enumerate(TECHNIQUES.items(), 1):
        print(f"  {i}. {technique.name} ({key})")
    
    try:
        choice = int(input("\nSelect technique number: ").strip())
        technique_keys = list(TECHNIQUES.keys())
        if 1 <= choice <= len(technique_keys):
            technique_key = technique_keys[choice - 1]
            technique = get_technique(technique_key)
        else:
            print("Invalid choice. Using Zero-Shot.")
            technique = get_technique("zero_shot")
    except (ValueError, IndexError):
        print("Invalid choice. Using Zero-Shot.")
        technique = get_technique("zero_shot")
    
    # Build and evaluate prompt
    print("\n" + "-" * 60)
    print("Building prompt...")
    prompt = technique.build_prompt(task)
    
    print("\nGenerated Prompt:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)
    
    # Execute
    print("\nExecuting prompt...")
    result = evaluator.evaluate(technique.name, prompt, temperature=0.7, max_tokens=1000)
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Execution Time: {result.execution_time:.2f}s")
    if result.token_count:
        print(f"Token Count: {result.token_count}")
    print(f"\nResponse:\n{result.response}")


def main():
    parser = argparse.ArgumentParser(
        description="Prompt Engineering Toolkit - Design, test, and optimize prompts"
    )
    parser.add_argument(
        "--example",
        choices=["math", "text", "react", "comprehensive"],
        help="Run a specific example"
    )
    parser.add_argument(
        "--list-techniques",
        action="store_true",
        help="List all available prompt techniques"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    if args.list_techniques:
        list_techniques()
    elif args.example:
        run_example(args.example)
    elif args.interactive:
        interactive_mode()
    else:
        # Default: show help and list techniques
        parser.print_help()
        print("\n")
        list_techniques()
        print("\n💡 Tip: Use --example to run examples or --interactive for interactive mode")


if __name__ == "__main__":
    main()

