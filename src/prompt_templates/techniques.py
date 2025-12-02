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
            prompt_parts.append("Eres un asistente experto. Por favor, proporciona una respuesta clara, precisa y bien razonada a la siguiente tarea.")
        
        prompt_parts.append("\n" + "-"*60)
        prompt_parts.append("TAREA")
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
            "Eres un experto en esta tarea. Estudia los ejemplos a continuación para entender el patrón y formato."
        )
        
        if instruction:
            prompt_parts.append(instruction)
        else:
            prompt_parts.append(default_instruction)
        
        if examples:
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("EJEMPLOS FEW-SHOT")
            prompt_parts.append("="*60)
            prompt_parts.append("\nEstos ejemplos demuestran el formato y razonamiento esperado:")
            
            for i, example in enumerate(examples, 1):
                input_text = example.get("input", "")
                output_text = example.get("output", "")
                reasoning = example.get("reasoning", "")
                
                prompt_parts.append(f"\n--- Ejemplo {i} ---")
                prompt_parts.append(f"Entrada: {input_text}")
                if reasoning:
                    prompt_parts.append(f"Razonamiento: {reasoning}")
                prompt_parts.append(f"Salida: {output_text}")
                prompt_parts.append("-" * 60)
            
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("TU TAREA")
            prompt_parts.append("="*60)
            prompt_parts.append(f"\nEntrada: {task}")
            prompt_parts.append("\nBasándote en los ejemplos anteriores, proporciona tu respuesta siguiendo el mismo patrón:")
            prompt_parts.append("Salida: [Tu respuesta aquí]")
        else:
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("NOTA")
            prompt_parts.append("="*60)
            prompt_parts.append("\nEste es un prompt few-shot. Para mejores resultados, proporciona 2-5 ejemplos que demuestren:")
            prompt_parts.append("  - El formato de entrada")
            prompt_parts.append("  - El formato de salida esperado")
            prompt_parts.append("  - El patrón de razonamiento (si aplica)")
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("TAREA")
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
            "Eres un experto en resolución de problemas. Al resolver problemas, debes:\n"
            "1. Dividir el problema en pasos más pequeños y manejables\n"
            "2. Trabajar cada paso metódicamente\n"
            "3. Mostrar tu razonamiento claramente en cada etapa\n"
            "4. Verificar tu solución antes de presentar la respuesta final\n\n"
            "Usa el siguiente formato para tu respuesta:\n"
            "Paso 1: [Identifica qué necesita resolverse]\n"
            "Paso 2: [Divide en componentes]\n"
            "Paso 3: [Aplica razonamiento/lógica]\n"
            "...\n"
            "Respuesta Final: [Tu conclusión]"
        )
        
        prompt_parts = []
        
        if instruction:
            prompt_parts.append(instruction)
        else:
            prompt_parts.append(default_instruction)
        
        if examples:
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("EJEMPLOS DE RAZONAMIENTO CADENA DE PENSAMIENTO")
            prompt_parts.append("="*60)
            for i, example in enumerate(examples, 1):
                input_text = example.get("input", "")
                reasoning = example.get("reasoning", "")
                output_text = example.get("output", "")
                prompt_parts.append(f"\nEjemplo {i}:")
                prompt_parts.append(f"Problema: {input_text}")
                prompt_parts.append(f"\nRazonamiento:")
                # Format reasoning with step indicators if not already formatted
                if "Paso" not in reasoning and "paso" not in reasoning and "Step" not in reasoning and "step" not in reasoning:
                    reasoning_lines = reasoning.split('\n')
                    formatted_reasoning = []
                    step_num = 1
                    for line in reasoning_lines:
                        if line.strip():
                            formatted_reasoning.append(f"Paso {step_num}: {line.strip()}")
                            step_num += 1
                    reasoning = '\n'.join(formatted_reasoning)
                prompt_parts.append(reasoning)
                prompt_parts.append(f"\nRespuesta: {output_text}")
                prompt_parts.append("-"*60)
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("TU TAREA")
        prompt_parts.append("="*60)
        prompt_parts.append(f"\n{task}")
        
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
        instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """Build a ReAct prompt following the standard Thought-Action-Observation loop format"""
        prompt_parts = []
        
        # Use custom instruction if provided, otherwise use default
        if instruction:
            prompt_parts.append(instruction)
        else:
            prompt_parts.extend([
                "Eres un asistente de IA que usa el framework ReAct (Razonamiento + Acción) para resolver problemas.",
                "ReAct combina razonamiento y acción en un bucle iterativo:",
                "",
                "1. PENSAMIENTO: Analiza la situación actual y determina qué información necesitas",
                "2. ACCIÓN: Realiza una acción específica para recopilar información o avanzar",
                "3. OBSERVACIÓN: Observa el resultado de tu acción",
                "4. Repite hasta que tengas suficiente información para proporcionar una respuesta final",
                "",
                "Debes alternar entre pasos de Pensamiento y Acción. Solo proporciona una Respuesta Final cuando tengas",
                "suficiente información para resolver el problema completamente."
            ])
        
        if available_actions:
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("ACCIONES DISPONIBLES")
            prompt_parts.append("="*60)
            for i, action in enumerate(available_actions, 1):
                prompt_parts.append(f"{i}. {action}")
            prompt_parts.append("")
            prompt_parts.append("Al realizar una acción, usa el formato: Acción: [nombre_accion](parámetros)")
        
        if examples:
            prompt_parts.append("\n" + "="*60)
            prompt_parts.append("EJEMPLOS REACT")
            prompt_parts.append("="*60)
            for i, example in enumerate(examples, 1):
                thought = example.get("thought", "")
                action = example.get("action", "")
                observation = example.get("observation", "")
                answer = example.get("answer", "")
                
                prompt_parts.append(f"\nEjemplo {i}:")
                prompt_parts.append(f"Problema: {example.get('problem', '')}")
                prompt_parts.append("")
                if thought:
                    prompt_parts.append(f"Pensamiento: {thought}")
                if action:
                    prompt_parts.append(f"Acción: {action}")
                if observation:
                    prompt_parts.append(f"Observación: {observation}")
                # Show multiple iterations if provided
                if example.get("thought2"):
                    prompt_parts.append(f"\nPensamiento: {example.get('thought2')}")
                    if example.get("action2"):
                        prompt_parts.append(f"Acción: {example.get('action2')}")
                    if example.get("observation2"):
                        prompt_parts.append(f"Observación: {example.get('observation2')}")
                if answer:
                    prompt_parts.append(f"\nRespuesta Final: {answer}")
                prompt_parts.append("-"*60)
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("TU TAREA")
        prompt_parts.append("="*60)
        prompt_parts.append(f"\n{task}")
        
        # ReAct needs format instructions for the response structure
        prompt_parts.append("\n" + "-"*60)
        prompt_parts.append("Formatea tu respuesta como:")
        prompt_parts.append("Pensamiento: [tu razonamiento]")
        prompt_parts.append("Acción: [acción a realizar]")
        prompt_parts.append("Observación: [resultado]")
        prompt_parts.append("... (repite según sea necesario)")
        prompt_parts.append("Respuesta Final: [tu solución]")
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
            "Estás resolviendo un problema usando el método de Auto-Consistencia. Esta técnica involucra:\n"
            "1. Generar múltiples caminos de razonamiento independientes para resolver el problema\n"
            "2. Cada camino debe usar un enfoque o perspectiva diferente\n"
            "3. Comparar las conclusiones de cada camino\n"
            "4. Seleccionar la respuesta que aparece más consistentemente entre los caminos\n\n"
            "Este método mejora la precisión al reducir el impacto de errores de razonamiento en cualquier camino individual."
        )
        
        prompt_parts = []
        
        if instruction:
            prompt_parts.append(instruction)
        else:
            prompt_parts.append(default_instruction)
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("TAREA")
        prompt_parts.append("="*60)
        prompt_parts.append(f"\n{task}")
        
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
            "Estás resolviendo un problema usando el método de Árbol de Pensamientos. Esta técnica involucra:\n"
            "1. Generar múltiples enfoques iniciales (ramas)\n"
            "2. Explorar cada rama en profundidad\n"
            "3. Evaluar el potencial de cada rama\n"
            "4. Podar las ramas menos prometedoras\n"
            "5. Seleccionar el camino más prometedor hacia la solución\n\n"
            "Piensa en esto como explorar un árbol de decisión donde cada rama representa una estrategia diferente."
        )
        
        prompt_parts = []
        
        if instruction:
            prompt_parts.append(instruction)
        else:
            prompt_parts.append(default_instruction)
        
        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("TAREA")
        prompt_parts.append("="*60)
        prompt_parts.append(f"\n{task}")
        
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

