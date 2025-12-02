"""
Benchmark suite for evaluating prompt engineering techniques
"""

from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.evaluator import PromptEvaluator, ComparisonResult
from src.prompt_templates.techniques import get_technique


class BenchmarkSuite:
    """Suite of benchmark tasks for evaluating prompt techniques"""
    
    def __init__(self, llm_client):
        """
        Initialize the benchmark suite with an LLM client.
        
        Args:
            llm_client: An LLM client instance (from src.utils.llm_client)
        """
        self.client = llm_client
        self.evaluator = PromptEvaluator(llm_client)
        self.tasks = self._initialize_tasks()
    
    def _initialize_tasks(self) -> List[Dict[str, Any]]:
        """Initialize the predefined benchmark tasks"""
        return [
            {
                "name": "Math Problem",
                "category": "Razonamiento matemático",
                "task": "Si un tren viaja 300 millas en 4 horas, y luego viaja otras 200 millas en 3 horas, ¿cuál es la velocidad promedio para todo el viaje?",
                "expected_keywords": ["velocidad", "promedio", "distancia", "tiempo"]
            },
            {
                "name": "Logical Reasoning",
                "category": "Razonamiento lógico",
                "task": "Todas las rosas son flores. Algunas flores se marchitan rápidamente. Por lo tanto, algunas rosas se marchitan rápidamente. ¿Es válido este razonamiento? Explica por qué sí o por qué no.",
                "expected_keywords": ["razonamiento", "válido", "lógica", "premisa"]
            },
            {
                "name": "Text Classification",
                "category": "Clasificación de texto",
                "task": "Analiza el sentimiento de este texto: '¡Acabo de recibir un ascenso y estoy tan emocionado por las nuevas oportunidades!'",
                "expected_keywords": ["sentimiento", "positivo", "negativo", "análisis"]
            },
            {
                "name": "Problem Solving",
                "category": "Resolución de problemas",
                "task": "Un restaurante tiene 50 mesas. Cada mesa puede acomodar 4 personas. Si el restaurante está lleno al 75% de su capacidad, ¿cuántas personas están comiendo en el restaurante?",
                "expected_keywords": ["mesas", "personas", "capacidad", "porcentaje"]
            }
        ]
    
    def run_benchmark(
        self, 
        technique_names: List[str],
        parallel_tasks: bool = True,
        max_workers: int = None
    ) -> List[ComparisonResult]:
        """
        Run benchmark tests with specified techniques on all tasks.
        
        Args:
            technique_names: List of technique names to test (e.g., ["zero_shot", "chain_of_thought"])
            parallel_tasks: If True, execute tasks in parallel (default: True)
            max_workers: Maximum number of parallel workers for tasks (default: None = auto)
        
        Returns:
            List of ComparisonResult objects, one for each task
        """
        def process_task(task_data: Dict[str, Any]) -> ComparisonResult:
            """Process a single task with all techniques"""
            task = task_data["task"]
            techniques = {}
            
            # Build prompts for each technique
            for tech_name in technique_names:
                try:
                    technique = get_technique(tech_name)
                    # Use default instructions from each technique
                    prompt = technique.build_prompt(task)
                    techniques[tech_name] = prompt
                except Exception as e:
                    # Skip techniques that fail to build
                    print(f"Warning: Could not build prompt for {tech_name}: {e}")
                    continue
            
            if not techniques:
                return None
            
            # Run comparison for this task (techniques already run in parallel internally)
            comparison = self.evaluator.compare_techniques(
                task=task,
                techniques=techniques,
                temperature=0.7,
                max_tokens=500,
                parallel=True  # Enable parallel execution of techniques
            )
            
            return comparison
        
        if parallel_tasks and len(self.tasks) > 1:
            # Execute tasks in parallel
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(process_task, task_data): i
                    for i, task_data in enumerate(self.tasks)
                }
                
                # Collect results maintaining order
                task_results = {}
                for future in as_completed(future_to_task):
                    task_idx = future_to_task[future]
                    try:
                        result = future.result()
                        if result is not None:
                            task_results[task_idx] = result
                    except Exception as e:
                        print(f"Error processing task {task_idx + 1}: {e}")
                
                # Reconstruct results in original task order
                results = [task_results[i] for i in sorted(task_results.keys())]
        else:
            # Execute tasks sequentially (original behavior)
            results = []
            for task_data in self.tasks:
                comparison = process_task(task_data)
                if comparison is not None:
                    results.append(comparison)
        
        return results

