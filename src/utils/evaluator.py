"""
Evaluation system for comparing prompt techniques
"""

from typing import List, Dict, Any, Optional, Callable
import time
import tiktoken
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EvaluationResult:
    """Result of a single prompt evaluation"""
    technique: str
    prompt: str
    response: str
    execution_time: float
    token_count: Optional[int] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ComparisonResult:
    """Results of comparing multiple techniques"""
    task: str
    results: List[EvaluationResult]
    best_technique: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class PromptEvaluator:
    """Evaluator for comparing different prompt techniques"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text"""
        try:
            return len(self.encoding.encode(text))
        except:
            # Fallback approximation
            return len(text.split()) * 1.3
    
    def evaluate(
        self,
        technique_name: str,
        prompt: str,
        scorer: Optional[Callable[[str], float]] = None,
        **kwargs
    ) -> EvaluationResult:
        """Evaluate a single prompt"""
        start_time = time.time()
        
        try:
            response = self.llm_client.generate(prompt, **kwargs)
            execution_time = time.time() - start_time
            
            token_count = self.count_tokens(prompt + response)
            score = None
            
            if scorer:
                score = scorer(response)
            
            return EvaluationResult(
                technique=technique_name,
                prompt=prompt,
                response=response,
                execution_time=execution_time,
                token_count=token_count,
                score=score,
                metadata=kwargs
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return EvaluationResult(
                technique=technique_name,
                prompt=prompt,
                response=f"Error: {str(e)}",
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
    
    def compare_techniques(
        self,
        task: str,
        techniques: Dict[str, str],  # technique_name -> prompt
        scorer: Optional[Callable[[str], float]] = None,
        **kwargs
    ) -> ComparisonResult:
        """Compare multiple prompt techniques on the same task"""
        results = []
        
        for technique_name, prompt in techniques.items():
            result = self.evaluate(technique_name, prompt, scorer, **kwargs)
            results.append(result)
        
        # Determine best technique (if scorer provided)
        best_technique = None
        if scorer:
            best_result = max(results, key=lambda r: r.score if r.score else -1)
            best_technique = best_result.technique
        
        return ComparisonResult(
            task=task,
            results=results,
            best_technique=best_technique
        )
    
    def generate_report(self, comparison_result: ComparisonResult) -> str:
        """Generate a human-readable report"""
        report = []
        report.append("=" * 80)
        report.append("PROMPT ENGINEERING EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nTask: {comparison_result.task}")
        report.append(f"Timestamp: {comparison_result.timestamp}")
        report.append("\n" + "-" * 80)
        
        for result in comparison_result.results:
            report.append(f"\nTechnique: {result.technique}")
            report.append(f"Execution Time: {result.execution_time:.2f}s")
            if result.token_count:
                report.append(f"Token Count: {result.token_count}")
            if result.score is not None:
                report.append(f"Score: {result.score:.2f}")
            report.append(f"\nPrompt:\n{result.prompt[:200]}...")
            report.append(f"\nResponse:\n{result.response[:500]}...")
            report.append("\n" + "-" * 80)
        
        if comparison_result.best_technique:
            report.append(f"\nBest Technique: {comparison_result.best_technique}")
        
        return "\n".join(report)

