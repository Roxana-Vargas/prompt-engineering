# Prompt Engineering Toolkit

Un conjunto de herramientas para diseñar, probar y optimizar prompts usando diferentes técnicas de prompt engineering.

## 🎥 Demo del proyecto

<a href="https://youtu.be/sOLWpS_XFWA" target="_blank">
  <img src="https://raw.githubusercontent.com/Roxana-Vargas/prompt-engineering/refs/heads/main/Captura%20de%20pantalla%202025-12-01%20212744.png" 
       alt="Demo" 
       style="width:100%; max-width:800px; border-radius:12px;">
</a>

## 🎯 Características

- **Múltiples Técnicas de Prompting**: Implementación de técnicas avanzadas como:
  - Zero-Shot Prompting
  - Few-Shot Prompting
  - Chain-of-Thought (CoT)
  - ReAct (Reasoning + Acting)
  - Self-Consistency
  - Tree-of-Thoughts

- **Sistema de Evaluación**: Sistema completo para comparar y evaluar diferentes técnicas de prompting
- **Métricas de Rendimiento**: Análisis de tiempo de ejecución, tokens utilizados y scores de calidad
- **Soporte Multi-Provider**: Compatible con OpenAI y Anthropic APIs
- **Ejemplos Prácticos**: Casos de uso reales demostrando cada técnica
- **🆕 Dashboard Interactivo**: Aplicación Streamlit con visualizaciones interactivas y gráficos

## 📋 Requisitos

- Python 3.8+
- API keys de OpenAI o Anthropic

## 🚀 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/Roxana-Vargas/prompt-engineering
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Configura las variables de entorno:
```bash
cp .env.example .env
# Edita .env y agrega tus API keys
```

## 🚀 Inicio rápido

### Opción 1: Aplicación Web con Streamlit (Recomendado) 🎨

```bash
# Instalar dependencias (incluye Streamlit)
pip install -r requirements.txt

# Ejecutar aplicación web
streamlit run streamlit_app.py

# O en Windows:
run_streamlit.bat
```

La aplicación web te permite:
- 🎯 Ejecutar benchmarks de forma interactiva
- 📊 Visualizar resultados con gráficos interactivos
- 📈 Comparar técnicas visualmente
- 📥 Descargar resultados en CSV

### Opción 2: Línea de Comandos

```bash
# Modo interactivo
python main.py --interactive

# Ejecutar ejemplos específicos
python main.py --example math
python main.py --example comprehensive

# Ver todas las técnicas disponibles
python main.py --list-techniques

```

## 💻 Uso básico

### Ejemplo 1: Comparación de técnicas

```python
from src.prompt_templates.techniques import get_technique
from src.utils.llm_client import get_client
from src.utils.evaluator import PromptEvaluator

# Inicializar cliente
client = get_client("openai", model="gpt-4")
evaluator = PromptEvaluator(client)

# Definir tarea
task = "Resuelve: Si tengo 5 manzanas y como 2, ¿cuántas me quedan?"

# Construir prompts con diferentes técnicas
zero_shot = get_technique("zero_shot")
cot = get_technique("chain_of_thought")

techniques = {
    "Zero-Shot": zero_shot.build_prompt(task),
    "Chain-of-Thought": cot.build_prompt(task)
}

# Comparar técnicas
comparison = evaluator.compare_techniques(
    task=task,
    techniques=techniques,
    temperature=0.7
)

# Generar reporte
report = evaluator.generate_report(comparison)
print(report)
```

### Ejemplo 2: Chain-of-Thought para Razonamiento Matemático

```python
from examples.math_reasoning import main
main()
```

### Ejemplo 3: Few-Shot Learning para Análisis de Texto

```python
from examples.text_analysis import main
main()
```

## 📚 Técnicas Implementadas

### Zero-Shot Prompting
Prompt directo sin ejemplos. Útil para tareas simples donde el modelo tiene conocimiento previo.

### Few-Shot Prompting
Incluye ejemplos de demostración para guiar al modelo. Mejora el rendimiento en tareas específicas.

### Chain-of-Thought (CoT)
Fomenta el razonamiento paso a paso. Especialmente efectivo para problemas matemáticos y lógicos.

### ReAct (Reasoning + Acting)
Combina razonamiento y acciones en un proceso iterativo. Ideal para tareas que requieren múltiples pasos.

### Self-Consistency
Genera múltiples caminos de razonamiento y selecciona la respuesta más consistente.

### Tree-of-Thoughts
Explora múltiples estrategias de razonamiento en una estructura de árbol.

## 📊 Estructura del proyecto

```
prompt-engineering/
├── src/
│   ├── prompt_templates/
│   │   └── techniques.py      # Implementación de técnicas
│   └── utils/
│       ├── config.py           # Configuración
│       ├── llm_client.py       # Clientes LLM
│       └── evaluator.py        # Sistema de evaluación
├── examples/
│   ├── math_reasoning.py       # Ejemplo: Razonamiento matemático
│   ├── text_analysis.py        # Ejemplo: Análisis de texto
│   ├── react_example.py        # Ejemplo: ReAct
│   └── comprehensive_comparison.py  # Comparación completa
├── streamlit_app.py            # 🆕 Dashboard interactivo Streamlit
├── main.py                     # Punto de entrada principal
├── requirements.txt
├── .env.example
├── run_streamlit.bat          # Script para ejecutar Streamlit (Windows)
├── run_streamlit.sh           # Script para ejecutar Streamlit (Linux/Mac)
└── README.md
```

## 🎓 Casos de uso

1. **Razonamiento Matemático**: Comparación de Zero-Shot vs Chain-of-Thought
2. **Análisis de Sentimiento**: Few-Shot learning para clasificación de texto
3. **Resolución de Problemas**: ReAct para tareas que requieren razonamiento iterativo
4. **Razonamiento Lógico**: Evaluación completa de múltiples técnicas

## 📈 Métricas de Evaluación

El toolkit incluye:
- **Tiempo de ejecución**: Medición del tiempo de respuesta
- **Conteo de tokens**: Análisis de eficiencia
- **Scores personalizados**: Métricas específicas por tarea
- **Reportes comparativos**: Análisis detallado de rendimiento

## 🎨 Dashboard Interactivo con Streamlit

El proyecto incluye una aplicación web interactiva construida con Streamlit que permite:

### Características del Dashboard:
- 📊 **Visualizaciones Interactivas**: Gráficos de barras comparativos usando Plotly
- ⏱️ **Métricas en Tiempo Real**: Tiempo de ejecución, tokens y scores
- 🔄 **Ejecución de Benchmarks**: Ejecuta benchmarks directamente desde la interfaz
- 📋 **Tablas Filtrables**: Filtra resultados por técnica o tarea
- 📥 **Exportación de Datos**: Descarga resultados en formato CSV
- 🎯 **Comparaciones Visuales**: Compara múltiples técnicas lado a lado

### Cómo usar el Dashboard:

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **En el navegador**:
   - Selecciona las técnicas a evaluar en la barra lateral
   - Haz clic en "Ejecutar Benchmarks"
   - Explora los gráficos y tablas interactivas
   - Descarga los resultados si lo necesitas

### Capturas de pantalla del Dashboard:
- Gráficos de tiempo de ejecución por técnica
- Comparación de uso de tokens
- Tablas detalladas con filtros
- Métricas resumidas en tarjetas

## 🔧 Configuración Avanzada

Puedes personalizar el comportamiento modificando `src/utils/config.py` o usando variables de entorno:

```python
from src.utils.config import Config

Config.DEFAULT_MODEL = "gpt-4-turbo"
Config.DEFAULT_TEMPERATURE = 0.5
Config.DEFAULT_MAX_TOKENS = 2000
```
## 🌐 Despliegue

Este proyecto puede desplegarse fácilmente en varias plataformas:

### Opción Rápida: Streamlit Cloud (Recomendado) ⭐

1. Sube tu código a GitHub
2. Ve a [Streamlit Cloud](https://streamlit.io/cloud)
3. Conecta tu repositorio
4. Configura tus API keys en Settings > Secrets
5. ¡Despliega! Tu app estará en `https://tu-app.streamlit.app`


📖 **Guía completa de despliegue**: Ver [DEPLOYMENT.md](DEPLOYMENT.md)



