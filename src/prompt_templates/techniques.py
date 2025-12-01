"""
Prompt templates for different prompt engineering techniques
"""

from typing import List, Dict, Any, Optional


class PromptTechnique:
    """Base class for prompt techniques"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def build_prompt(self, task: str, **kwargs) -> str:
        """Build a prompt for the given task"""
        raise NotImplementedError


class ZeroShotPrompt(PromptTechnique):
    """Zero-shot prompting - no examples provided"""
    
    def __init__(self):
        super().__init__(
            name="Zero-Shot",
            description="Direct prompt without examples"
        )
    
    def build_prompt(self, task: str, instruction: Optional[str] = None, **kwargs) -> str:
        """Build a zero-shot prompt with clear structure"""
        prompt_parts = []
        
        if instruction:
            prompt_parts.append(instruction)
        else:
            # Provide a professional default structure
            prompt_parts.append("You are an expert assistant. Please provide a clear, accurate, and well-reasoned response to the following task.")
        
        prompt_parts.append("\n" + "-"*60)
        prompt_parts.append("TASK")
        prompt_parts.append("-"*60)
        prompt_parts.append(f"\n{task}")
        prompt_parts.append("\n" + "-"*60)
        
        return "\n".join(prompt_parts)


class FewShotPrompt(PromptTechnique):
    """Few-shot prompting - includes examples"""
    
    def __init__(self):
        super().__init__(
            name="Few-Shot",
            description="Prompt with example demonstrations"
        )
    
    def build_prompt(
        self,
        task: str,
        examples: Optional[List[Dict[str, str]]] = None,
        instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Build a few-shot prompt with well-structured examples"""
        prompt_parts = []
        
        default_instruction = (
            "You are an expert at this task. Study the examples below to understand the pattern and format."
        )
        
        if instruction:
            prompt_parts.append(instruction)
        else:
            prompt_parts.append(default_instruction)
        
        if examples:
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("FEW-SHOT EXAMPLES")
            prompt_parts.append("="*60)
            prompt_parts.append("\nThese examples demonstrate the expected format and reasoning:")
            
            for i, example in enumerate(examples, 1):
                input_text = example.get("input", "")
                output_text = example.get("output", "")
                reasoning = example.get("reasoning", "")
                
                prompt_parts.append(f"\n--- Example {i} ---")
                prompt_parts.append(f"Input: {input_text}")
                if reasoning:
                    prompt_parts.append(f"Reasoning: {reasoning}")
                prompt_parts.append(f"Output: {output_text}")
                prompt_parts.append("-" * 60)
            
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("YOUR TASK")
            prompt_parts.append("="*60)
            prompt_parts.append(f"\nInput: {task}")
            prompt_parts.append("\nBased on the examples above, provide your response following the same pattern:")
            prompt_parts.append("Output: [Your response here]")
        else:
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("NOTE")
            prompt_parts.append("="*60)
            prompt_parts.append("\nThis is a few-shot prompt. For best results, provide 2-5 examples that demonstrate:")
            prompt_parts.append("  - The input format")
            prompt_parts.append("  - The expected output format")
            prompt_parts.append("  - The reasoning pattern (if applicable)")
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("TASK")
            prompt_parts.append("="*60)
            prompt_parts.append(f"\n{task}")
        
        return "\n".join(prompt_parts)


class ChainOfThoughtPrompt(PromptTechnique):
    """Chain-of-Thought (CoT) prompting - encourages step-by-step reasoning"""
    
    def __init__(self):
        super().__init__(
            name="Chain-of-Thought",
            description="Encourages step-by-step reasoning process"
        )
    
    def build_prompt(
        self,
        task: str,
        instruction: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Build a Chain-of-Thought prompt with structured reasoning format"""
        default_instruction = (
            "You are an expert problem solver. When solving problems, you should:\n"
            "1. Break down the problem into smaller, manageable steps\n"
            "2. Work through each step methodically\n"
            "3. Show your reasoning clearly at each stage\n"
            "4. Verify your solution before presenting the final answer\n\n"
            "Use the following format for your response:\n"
            "Step 1: [Identify what needs to be solved]\n"
            "Step 2: [Break down into components]\n"
            "Step 3: [Apply reasoning/logic]\n"
            "...\n"
            "Final Answer: [Your conclusion]"
        )
        
        prompt_parts = []
        
        if instruction:
            prompt_parts.append(instruction)
        else:
            prompt_parts.append(default_instruction)
        
        if examples:
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("EXAMPLES OF CHAIN-OF-THOUGHT REASONING")
            prompt_parts.append("="*60)
            for i, example in enumerate(examples, 1):
                input_text = example.get("input", "")
                reasoning = example.get("reasoning", "")
                output_text = example.get("output", "")
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Problem: {input_text}")
                prompt_parts.append(f"\nReasoning:")
                # Format reasoning with step indicators if not already formatted
                if "Step" not in reasoning and "step" not in reasoning:
                    reasoning_lines = reasoning.split('\n')
                    formatted_reasoning = []
                    step_num = 1
                    for line in reasoning_lines:
                        if line.strip():
                            formatted_reasoning.append(f"Step {step_num}: {line.strip()}")
                            step_num += 1
                    reasoning = '\n'.join(formatted_reasoning)
                prompt_parts.append(reasoning)
                prompt_parts.append(f"\nAnswer: {output_text}")
                prompt_parts.append("-"*60)
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("YOUR TASK")
        prompt_parts.append("="*60)
        prompt_parts.append(f"\n{task}")
        prompt_parts.append("\n" + "-"*60)
        prompt_parts.append("Now, solve this problem using the Chain-of-Thought method.")
        prompt_parts.append("Show each step of your reasoning clearly and methodically.")
        prompt_parts.append("-"*60)
        
        return "\n".join(prompt_parts)


class ReActPrompt(PromptTechnique):
    """ReAct (Reasoning + Acting) prompting - combines reasoning and actions"""
    
    def __init__(self):
        super().__init__(
            name="ReAct",
            description="Combines reasoning and acting in an iterative process"
        )
    
    def build_prompt(
        self,
        task: str,
        available_actions: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Build a ReAct prompt following the standard Thought-Action-Observation loop format"""
        prompt_parts = [
            "You are an AI assistant that uses the ReAct (Reasoning + Acting) framework to solve problems.",
            "ReAct combines reasoning and acting in an iterative loop:",
            "",
            "1. THOUGHT: Analyze the current situation and determine what information you need",
            "2. ACTION: Take a specific action to gather information or make progress",
            "3. OBSERVATION: Observe the result of your action",
            "4. Repeat until you have enough information to provide a final answer",
            "",
            "You must alternate between Thought and Action steps. Only provide a Final Answer when you have",
            "sufficient information to solve the problem completely."
        ]
        
        if available_actions:
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("AVAILABLE ACTIONS")
            prompt_parts.append("="*60)
            for i, action in enumerate(available_actions, 1):
                prompt_parts.append(f"{i}. {action}")
            prompt_parts.append("")
            prompt_parts.append("When taking an action, use the format: Action: [action_name](parameters)")
        
        if examples:
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("REACT EXAMPLES")
            prompt_parts.append("="*60)
            for i, example in enumerate(examples, 1):
                thought = example.get("thought", "")
                action = example.get("action", "")
                observation = example.get("observation", "")
                answer = example.get("answer", "")
                
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Problem: {example.get('problem', '')}")
                prompt_parts.append("")
                if thought:
                    prompt_parts.append(f"Thought: {thought}")
                if action:
                    prompt_parts.append(f"Action: {action}")
                if observation:
                    prompt_parts.append(f"Observation: {observation}")
                # Show multiple iterations if provided
                if example.get("thought2"):
                    prompt_parts.append(f"\nThought: {example.get('thought2')}")
                    if example.get("action2"):
                        prompt_parts.append(f"Action: {example.get('action2')}")
                    if example.get("observation2"):
                        prompt_parts.append(f"Observation: {example.get('observation2')}")
                if answer:
                    prompt_parts.append(f"\nFinal Answer: {answer}")
                prompt_parts.append("-"*60)
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("YOUR TASK")
        prompt_parts.append("="*60)
        prompt_parts.append(f"\n{task}")
        prompt_parts.append("\n" + "-"*60)
        prompt_parts.append("Solve this problem using the ReAct framework.")
        prompt_parts.append("Start with a Thought, then take an Action, observe the result, and continue.")
        prompt_parts.append("Format your response as:")
        prompt_parts.append("Thought: [your reasoning]")
        prompt_parts.append("Action: [action to take]")
        prompt_parts.append("Observation: [result]")
        prompt_parts.append("... (repeat as needed)")
        prompt_parts.append("Final Answer: [your solution]")
        prompt_parts.append("-"*60)
        
        return "\n".join(prompt_parts)


class SelfConsistencyPrompt(PromptTechnique):
    """Self-Consistency prompting - generates multiple reasoning paths"""
    
    def __init__(self):
        super().__init__(
            name="Self-Consistency",
            description="Generates multiple reasoning paths and selects the most consistent answer"
        )
    
    def build_prompt(
        self,
        task: str,
        instruction: Optional[str] = None,
        num_paths: int = 3,
        **kwargs
    ) -> str:
        """Build a self-consistency prompt that encourages multiple reasoning paths"""
        default_instruction = (
            "You are solving a problem using the Self-Consistency method. This technique involves:\n"
            "1. Generating multiple independent reasoning paths to solve the problem\n"
            "2. Each path should use a different approach or perspective\n"
            "3. Comparing the conclusions from each path\n"
            "4. Selecting the answer that appears most consistently across paths\n\n"
            "This method improves accuracy by reducing the impact of reasoning errors in any single path."
        )
        
        prompt_parts = []
        
        if instruction:
            prompt_parts.append(instruction)
        else:
            prompt_parts.append(default_instruction)
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("TASK")
        prompt_parts.append("="*60)
        prompt_parts.append(f"\n{task}")
        prompt_parts.append("\n" + "-"*60)
        prompt_parts.append("INSTRUCTIONS")
        prompt_parts.append("-"*60)
        prompt_parts.append(f"\nGenerate {num_paths} different reasoning paths to solve this problem.")
        prompt_parts.append("Each path should:")
        prompt_parts.append("  - Use a different approach or starting point")
        prompt_parts.append("  - Show complete reasoning from start to finish")
        prompt_parts.append("  - Arrive at a conclusion")
        prompt_parts.append("\nAfter showing all paths, compare them and identify the most consistent answer.")
        prompt_parts.append("\nFormat your response as:")
        prompt_parts.append("\n---")
        prompt_parts.append("REASONING PATH 1:")
        prompt_parts.append("[Approach: describe your approach]")
        prompt_parts.append("[Show your reasoning step by step]")
        prompt_parts.append("Conclusion: [your answer]")
        prompt_parts.append("\n---")
        prompt_parts.append("REASONING PATH 2:")
        prompt_parts.append("[Different approach]")
        prompt_parts.append("[Show reasoning]")
        prompt_parts.append("Conclusion: [your answer]")
        prompt_parts.append("\n---")
        prompt_parts.append("REASONING PATH 3:")
        prompt_parts.append("[Another different approach]")
        prompt_parts.append("[Show reasoning]")
        prompt_parts.append("Conclusion: [your answer]")
        prompt_parts.append("\n---")
        prompt_parts.append("SELF-CONSISTENCY ANALYSIS:")
        prompt_parts.append("Comparing the conclusions from all paths...")
        prompt_parts.append("Most consistent answer: [your final answer]")
        prompt_parts.append("---")
        
        return "\n".join(prompt_parts)


class TreeOfThoughtsPrompt(PromptTechnique):
    """Tree of Thoughts - explores multiple reasoning paths"""
    
    def __init__(self):
        super().__init__(
            name="Tree-of-Thoughts",
            description="Explores multiple reasoning paths in a tree structure"
        )
    
    def build_prompt(
        self,
        task: str,
        instruction: Optional[str] = None,
        num_branches: int = 3,
        **kwargs
    ) -> str:
        """Build a Tree of Thoughts prompt with structured exploration"""
        default_instruction = (
            "You are solving a problem using the Tree of Thoughts method. This technique involves:\n"
            "1. Generating multiple initial approaches (branches)\n"
            "2. Exploring each branch in depth\n"
            "3. Evaluating the potential of each branch\n"
            "4. Pruning less promising branches\n"
            "5. Selecting the most promising path to the solution\n\n"
            "Think of this as exploring a decision tree where each branch represents a different strategy."
        )
        
        prompt_parts = []
        
        if instruction:
            prompt_parts.append(instruction)
        else:
            prompt_parts.append(default_instruction)
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("TASK")
        prompt_parts.append("="*60)
        prompt_parts.append(f"\n{task}")
        prompt_parts.append("\n" + "-"*60)
        prompt_parts.append("TREE OF THOUGHTS EXPLORATION")
        prompt_parts.append("-"*60)
        prompt_parts.append(f"\nGenerate {num_branches} distinct approaches to solve this problem.")
        prompt_parts.append("For each approach, you should:")
        prompt_parts.append("  1. Describe the strategy")
        prompt_parts.append("  2. Explore how it would work in detail")
        prompt_parts.append("  3. Identify potential challenges or limitations")
        prompt_parts.append("  4. Evaluate its feasibility and effectiveness")
        prompt_parts.append("\nAfter exploring all branches, compare them and select the best path.")
        prompt_parts.append("\nFormat your response as:")
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("BRANCH 1: [Name of Approach]")
        prompt_parts.append("="*60)
        prompt_parts.append("Strategy: [Describe the approach]")
        prompt_parts.append("Exploration: [Work through how this approach would solve the problem]")
        prompt_parts.append("Challenges: [Identify potential issues]")
        prompt_parts.append("Evaluation: [Assess feasibility and effectiveness]")
        prompt_parts.append("Potential Outcome: [Expected result]")
        
        for i in range(2, num_branches + 1):
            prompt_parts.append(f"\n{'='*60}")
            prompt_parts.append(f"BRANCH {i}: [Different Approach Name]")
            prompt_parts.append("="*60)
            prompt_parts.append("Strategy: [Describe]")
            prompt_parts.append("Exploration: [Work through]")
            prompt_parts.append("Challenges: [Identify]")
            prompt_parts.append("Evaluation: [Assess]")
            prompt_parts.append("Potential Outcome: [Expected result]")
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("BRANCH COMPARISON & SELECTION")
        prompt_parts.append("="*60)
        prompt_parts.append("Comparing all branches:")
        prompt_parts.append("  - Strengths and weaknesses of each")
        prompt_parts.append("  - Likelihood of success")
        prompt_parts.append("  - Resource requirements")
        prompt_parts.append("\nSelected Branch: [Which approach you choose]")
        prompt_parts.append("Reasoning: [Why this branch is best]")
        prompt_parts.append("\n" + "-"*60)
        prompt_parts.append("FINAL SOLUTION")
        prompt_parts.append("-"*60)
        prompt_parts.append("Using the selected branch, provide the complete solution:")
        prompt_parts.append("[Your detailed solution using the chosen approach]")
        prompt_parts.append("="*60)
        
        return "\n".join(prompt_parts)


# Registry of all available techniques
TECHNIQUES = {
    "zero_shot": ZeroShotPrompt(),
    "few_shot": FewShotPrompt(),
    "chain_of_thought": ChainOfThoughtPrompt(),
    "react": ReActPrompt(),
    "self_consistency": SelfConsistencyPrompt(),
    "tree_of_thoughts": TreeOfThoughtsPrompt(),
}


def get_technique(name: str) -> PromptTechnique:
    """Get a prompt technique by name"""
    if name not in TECHNIQUES:
        raise ValueError(f"Unknown technique: {name}. Available: {list(TECHNIQUES.keys())}")
    return TECHNIQUES[name]

