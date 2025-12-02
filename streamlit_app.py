"""
Streamlit App for Prompt Engineering Benchmark Visualization
"""

# Standard library imports
import os
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Local imports
from benchmarks.benchmark_suite import BenchmarkSuite
from src.prompt_templates.techniques import get_technique
from src.utils.config import Config
from src.utils.evaluator import PromptEvaluator, ComparisonResult
from src.utils.llm_client import get_client


# Page configuration
st.set_page_config(
    page_title="Prompt Engineering Benchmark",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


def load_benchmark_results():
    """Load or run benchmark results"""
    if 'benchmark_results' not in st.session_state:
        st.session_state.benchmark_results = None
    
    return st.session_state.benchmark_results


def run_benchmarks():
    """Run benchmark suite"""
    try:
        with st.spinner("Ejecutando benchmarks en paralelo... Esto será mucho más rápido"):
            client = get_client("openai", model="gpt-4")
            suite = BenchmarkSuite(client)
            techniques_to_test = ["zero_shot", "chain_of_thought", "react", "self_consistency", "tree_of_thoughts"]
            results = suite.run_benchmark(techniques_to_test, parallel_tasks=True)
            st.session_state.benchmark_results = results
            st.success("✅ Benchmarks completados!")
            return results
    except Exception as e:
        st.error(f"Error ejecutando benchmarks: {str(e)}")
        st.info("💡 Asegúrate de tener configurada tu API key en el archivo .env")
        return None


def _build_technique_prompt(technique, tech_name: str, custom_task: str):
    """Build prompt for a specific technique with custom instructions."""
    return technique.build_prompt(custom_task)


def run_custom_task(client, custom_task: str, technique_names: list) -> list:
    """Run a custom task with specified techniques."""
    evaluator = PromptEvaluator(client)
    techniques = {}
    
    # Build prompts for each technique
    for tech_name in technique_names:
        technique = get_technique(tech_name)
        prompt = _build_technique_prompt(technique, tech_name, custom_task)
        techniques[tech_name] = prompt
    
    # Run comparison (with parallel execution enabled by default)
    comparison = evaluator.compare_techniques(
        task=custom_task,
        techniques=techniques,
        temperature=0.7,
        max_tokens=500,
        parallel=True  # Execute techniques in parallel
    )
    
    return [comparison]


def prepare_dataframe(results):
    """Convert benchmark results to DataFrame."""
    TASK_TEXT_MAX_LENGTH = 50
    data = []
    
    for i, comparison in enumerate(results):
        task_name = f"Task {i+1}"
        task_text = (
            comparison.task[:TASK_TEXT_MAX_LENGTH] + "..."
            if len(comparison.task) > TASK_TEXT_MAX_LENGTH
            else comparison.task
        )
        
        for result in comparison.results:
            data.append({
                "Task": task_name,
                "Task Text": task_text,
                "Technique": result.technique.replace("_", " ").title(),
                "Execution Time (s)": round(result.execution_time, 2),
                "Token Count": result.token_count if result.token_count is not None else 0,
                "Full Task": comparison.task
            })
    
    return pd.DataFrame(data)


def create_comparison_charts(df):
    """Create comparison charts."""
    CHART_HEIGHT = 400
    
    # Chart 1: Execution Time Comparison
    fig_time = px.bar(
        df,
        x="Task",
        y="Execution Time (s)",
        color="Technique",
        title="⏱️ Tiempo de ejecución por técnica",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_time.update_layout(
        height=CHART_HEIGHT,
        xaxis_title="Tarea",
        yaxis_title="Tiempo (segundos)",
        legend_title="Técnica"
    )
    
    # Chart 2: Token Usage Comparison
    fig_tokens = px.bar(
        df,
        x="Task",
        y="Token Count",
        color="Technique",
        title="💰 Uso de tokens por técnica",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_tokens.update_layout(
        height=CHART_HEIGHT,
        xaxis_title="Tarea",
        yaxis_title="Tokens",
        legend_title="Técnica"
    )
    
    # Chart 3: Average Performance
    avg_df = df.groupby("Technique").agg({
        "Execution Time (s)": "mean",
        "Token Count": "mean"
    }).reset_index()
    
    fig_avg = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Tiempo Promedio", "Tokens Promedio"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig_avg.add_trace(
        go.Bar(
            x=avg_df["Technique"],
            y=avg_df["Execution Time (s)"],
            name="Tiempo",
            marker_color="#1f77b4"
        ),
        row=1,
        col=1
    )
    
    fig_avg.add_trace(
        go.Bar(
            x=avg_df["Technique"],
            y=avg_df["Token Count"],
            name="Tokens",
            marker_color="#ff7f0e"
        ),
        row=1,
        col=2
    )
    
    fig_avg.update_layout(
        height=CHART_HEIGHT,
        title_text="📊 Rendimiento promedio por técnica",
        showlegend=False
    )
    
    return fig_time, fig_tokens, fig_avg


def check_api_keys():
    """Check if API keys are available and allow user to input them if not."""
    openai_key = Config.get_openai_key()
    anthropic_key = Config.get_anthropic_key()
    
    # Check session state for user-provided keys
    if 'user_openai_key' in st.session_state and st.session_state.user_openai_key:
        openai_key = st.session_state.user_openai_key
        # Set it as environment variable for this session
        os.environ['OPENAI_API_KEY'] = openai_key
    
    if 'user_anthropic_key' in st.session_state and st.session_state.user_anthropic_key:
        anthropic_key = st.session_state.user_anthropic_key
        # Set it as environment variable for this session
        os.environ['ANTHROPIC_API_KEY'] = anthropic_key
    
    return openai_key, anthropic_key


def _handle_api_key_input(key_name: str, display_name: str, placeholder: str, env_var: str):
    """Handle API key input for a specific provider."""
    current_key = st.session_state.get(f'user_{key_name}', '')
    config_key = Config.get_openai_key() if key_name == 'openai_key' else Config.get_anthropic_key()
    
    if not current_key and config_key:
        # Show masked version if loaded from .env
        masked_key = config_key[-4:] if len(config_key) > 4 else '****'
        st.caption(f"ℹ️ Cargada desde .env (...{masked_key})")
    
    key_input = st.text_input(
        display_name,
        type="password",
        value=current_key,
        placeholder=placeholder,
        help=f"Ingresa tu API key de {display_name}",
        key=f"{key_name}_input_sidebar"
    )
    
    # Save to session state and environment when user types
    if key_input and key_input != current_key:
        st.session_state[f'user_{key_name}'] = key_input
        os.environ[env_var] = key_input
        if key_input:
            st.success("✅ Guardada")


def show_api_key_input_in_sidebar():
    """Show API key input form in sidebar - always available for user to configure."""
    openai_key, anthropic_key = check_api_keys()
    
    # Check if at least one key is configured
    has_at_least_one_key = bool(openai_key or anthropic_key)
    
    st.subheader("🔑 API Keys")
    
    if not has_at_least_one_key:
        st.info("⚠️ Configura al menos una API key para continuar")
    
    # Handle OpenAI input
    _handle_api_key_input('openai_key', 'OpenAI API Key', 'sk-...', 'OPENAI_API_KEY')
    
    # Handle Anthropic input
    _handle_api_key_input('anthropic_key', 'Anthropic API Key', 'sk-ant-...', 'ANTHROPIC_API_KEY')
    
    st.caption("💡 Las keys se guardan solo para esta sesión")
    
    # Update has_at_least_one_key after checking user inputs
    openai_key, anthropic_key = check_api_keys()
    has_at_least_one_key = bool(openai_key or anthropic_key)
    
    # Return True if at least one key is configured
    return has_at_least_one_key


def _get_selected_techniques(zero_shot, chain_of_thought, react, self_consistency, tree_of_thoughts):
    """Build list of selected techniques from checkbox values."""
    techniques = []
    if zero_shot:
        techniques.append("zero_shot")
    if chain_of_thought:
        techniques.append("chain_of_thought")
    if react:
        techniques.append("react")
    if self_consistency:
        techniques.append("self_consistency")
    if tree_of_thoughts:
        techniques.append("tree_of_thoughts")
    return techniques


def _execute_benchmarks(api_provider, task_mode, custom_task, techniques):
    """Execute benchmarks based on task mode and selected techniques."""
    MIN_TASK_LENGTH = 10
    openai_key, anthropic_key = check_api_keys()
    
    # Verify API key is available for selected provider
    if api_provider == "openai" and not openai_key:
        st.error("❌ Por favor, configura tu OpenAI API key primero")
        return False
    elif api_provider == "anthropic" and not anthropic_key:
        st.error("❌ Por favor, configura tu Anthropic API key primero")
        return False
    
    # Validate techniques
    if not techniques:
        st.warning("⚠️ Selecciona al menos una técnica")
        return False
    
    # Validate custom task if selected
    if task_mode == "Tarea Personalizada":
        if not custom_task or len(custom_task.strip()) < MIN_TASK_LENGTH:
            st.error(f"❌ Por favor, ingresa una tarea personalizada válida (mínimo {MIN_TASK_LENGTH} caracteres)")
            return False
    
    # Execute benchmarks
    try:
        client = get_client(api_provider, model="gpt-4")
        
        if task_mode == "Tarea Personalizada":
            with st.spinner("Ejecutando tarea personalizada en paralelo..."):
                results = run_custom_task(client, custom_task.strip(), techniques)
        else:
            with st.spinner("Ejecutando benchmarks en paralelo..."):
                suite = BenchmarkSuite(client)
                results = suite.run_benchmark(techniques, parallel_tasks=True)
        
        st.session_state.benchmark_results = results
        st.session_state.benchmark_df = prepare_dataframe(results)
        st.success("✅ Completado!")
        st.rerun()
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False


def _show_predefined_tasks_info():
    """Show information about predefined tasks."""
    DEFAULT_TASKS_INFO = """
    Esta aplicación evaluará las siguientes tareas:
    
    1. **Math Problem** - Razonamiento matemático
    2. **Logical Reasoning** - Razonamiento lógico
    3. **Text Classification** - Clasificación de texto
    4. **Problem Solving** - Resolución de problemas
    
    Configura tus API keys para ver los detalles completos de cada tarea.
    """
    return DEFAULT_TASKS_INFO


def _show_tasks_list(api_keys_configured):
    """Show list of tasks that will be executed."""
    custom_task = st.session_state.get('custom_task', '')
    task_mode = st.session_state.get('task_mode', 'Tareas Predefinidas')
    
    if task_mode == "Tarea Personalizada" and custom_task:
        st.markdown("### ✨ Tarea Personalizada")
        st.info(custom_task)
        st.caption("💡 Esta es tu tarea personalizada. Se evaluará con las técnicas seleccionadas.")
        return
    
    # Get tasks from BenchmarkSuite to show them
    try:
        if api_keys_configured:
            selected_provider = st.session_state.get('selected_provider', 'openai')
            try:
                temp_client = get_client(selected_provider, model="gpt-4")
                suite = BenchmarkSuite(temp_client)
            except Exception:
                suite = None
            
            if suite:
                st.markdown("### 📝 Lista de Tareas predefinidas")
                for i, task_data in enumerate(suite.tasks, 1):
                    with st.expander(
                        f"Tarea {i}: {task_data['name']} ({task_data['category']})",
                        expanded=False
                    ):
                        st.write("**Tarea:**")
                        st.info(task_data['task'])
                        st.write(f"**Categoría:** {task_data['category']}")
                        if task_data.get('expected_keywords'):
                            keywords = ', '.join(task_data['expected_keywords'])
                            st.write(f"**Palabras clave esperadas:** {keywords}")
            else:
                st.info(_show_predefined_tasks_info())
        else:
            st.info(_show_predefined_tasks_info())
    except Exception:
        st.info(_show_predefined_tasks_info())


def _show_initial_info(api_keys_configured):
    """Show initial information when no results are available."""
    st.subheader("📋 Tareas que se realizarán")
    _show_tasks_list(api_keys_configured)
    
    st.markdown("""
    ### 🎯 Técnicas disponibles
    
    - **Zero-Shot**: El modelo responde sin ejemplos previos
    - **Chain-of-Thought**: El modelo razona paso a paso
    - **ReAct**: Combina razonamiento y acciones en un proceso iterativo
    - **Self-Consistency**: Genera múltiples caminos de razonamiento y selecciona el más consistente
    - **Tree-of-Thoughts**: Explora múltiples caminos de razonamiento en una estructura de árbol
    
    ### 📊 Métricas que se evaluarán
    
    - ⏱️ **Tiempo de Ejecución**: Cuánto tarda cada técnica
    - 💰 **Uso de Tokens**: Cantidad de tokens consumidos
    
    ### 🚀 Cómo empezar
    
    1. Configura tus API keys en la barra lateral (←)
    2. Selecciona el proveedor (OpenAI o Anthropic)
    3. Elige las técnicas que deseas evaluar
    4. Haz clic en "🔄 Ejecutar Benchmarks"
    
    ### 💡 Obtener API Keys
    
    - **OpenAI**: Visita https://platform.openai.com/api-keys
    - **Anthropic**: Visita https://console.anthropic.com/
    """)
    
    if not api_keys_configured:
        st.warning("⚠️ Por favor, configura al menos una API key en la barra lateral para continuar")


def _show_results_summary(df, results):
    """Show summary metrics of benchmark results."""
    st.subheader("📊 Resumen de Resultados")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tareas", len(results))
    
    with col2:
        avg_time = df["Execution Time (s)"].mean()
        st.metric("Tiempo Promedio", f"{avg_time:.2f}s")
    
    with col3:
        total_tokens = df["Token Count"].sum()
        st.metric("Total Tokens", f"{total_tokens:,}")
    
    with col4:
        best_technique = df.groupby("Technique")["Execution Time (s)"].mean().idxmin()
        st.metric("Técnica Más Rápida", best_technique)


def _show_results_table(df):
    """Show detailed results table with filters."""
    st.subheader("📋 Tabla detallada de resultados")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        selected_techniques = st.multiselect(
            "Filtrar por Técnica",
            options=df["Technique"].unique(),
            default=df["Technique"].unique()
        )
    
    with col2:
        selected_tasks = st.multiselect(
            "Filtrar por Tarea",
            options=df["Task"].unique(),
            default=df["Task"].unique()
        )
    
    # Filter dataframe
    filtered_df = df[
        (df["Technique"].isin(selected_techniques)) &
        (df["Task"].isin(selected_tasks))
    ]
    
    # Display table
    st.dataframe(
        filtered_df[["Task", "Technique", "Execution Time (s)", "Token Count"]],
        width='stretch',
        hide_index=True
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Descargar CSV",
        data=csv,
        file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    return filtered_df


def _show_task_details(results, df):
    """Show detailed task information with prompts and responses."""
    TASK_PREVIEW_LENGTH = 60
    st.subheader("📝 Detalles de tareas y prompts enviados")
    
    for i, comparison in enumerate(results):
        task_preview = (
            comparison.task[:TASK_PREVIEW_LENGTH] + "..."
            if len(comparison.task) > TASK_PREVIEW_LENGTH
            else comparison.task
        )
        
        with st.expander(f"Tarea {i+1}: {task_preview}", expanded=False):
            st.write("**Tarea completa:**")
            st.info(comparison.task)
            st.markdown("---")
            
            # Show prompts and responses together for each technique
            st.write("**📤 Prompts y respuestas por técnica:**")
            for result in comparison.results:
                technique_display = result.technique.replace("_", " ").title()
                with st.expander(f"🔹 {technique_display}", expanded=False):
                    # Prompt section
                    st.write("**📤 Prompt enviado al modelo:**")
                    st.code(result.prompt, language="text")
                    st.caption(f"Tokens en prompt: ~{len(result.prompt.split())} palabras")
                    st.markdown("---")
                    
                    # Response section
                    st.write("**📥 Respuesta del Modelo:**")
                    st.write(result.response)
                    st.markdown("---")
                    
                    # Metrics
                    st.write("**📊 Métricas:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Tiempo", f"{result.execution_time:.2f}s")
                    with col2:
                        token_count = result.token_count if result.token_count is not None else 0
                        st.metric("Tokens", f"{token_count:,}")
            
            st.markdown("---")
            
            # Show results table
            st.write("**📊 Resumen de resultados:**")
            task_df = df[df["Task"] == f"Task {i+1}"]
            st.dataframe(
                task_df[["Technique", "Execution Time (s)", "Token Count"]],
                width='stretch',
                hide_index=True
            )


def main():
    """Main Streamlit app"""
    # Header
    st.markdown('<h1 class="main-header">🚀 Prompt Engineering Benchmark</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Show API key input in sidebar
        api_keys_configured = show_api_key_input_in_sidebar()
        
        st.markdown("---")
        
        st.subheader("⚙️ Configuración de Benchmarks")
        api_provider = st.selectbox(
            "Proveedor",
            ["openai", "anthropic"],
            index=0,
            key="provider_selectbox"
        )
        # Store in session state for access in main content
        st.session_state.selected_provider = api_provider
        
        # Check if selected provider has API key
        openai_key, anthropic_key = check_api_keys()
        if api_provider == "openai" and not openai_key:
            st.warning("⚠️ OpenAI API key no configurada")
        elif api_provider == "anthropic" and not anthropic_key:
            st.warning("⚠️ Anthropic API key no configurada")
        
        st.markdown("---")
        
        st.subheader("📝 Tipo de Tarea")
        task_mode = st.radio(
            "Selecciona el tipo de tarea:",
            ["Tareas Predefinidas", "Tarea Personalizada"],
            index=0,
            key="task_mode"
        )
        
        custom_task = None
        if task_mode == "Tarea Personalizada":
            custom_task = st.text_area(
                "Ingresa tu tarea personalizada:",
                value=st.session_state.get('custom_task', ''),
                placeholder="Ejemplo: Explica cómo funciona la fotosíntesis en las plantas.",
                help="Escribe la tarea que deseas evaluar con las diferentes técnicas de prompting",
                key="custom_task_input",
                height=100
            )
            st.session_state.custom_task = custom_task
            
            if custom_task and len(custom_task.strip()) < 10:
                st.warning("⚠️ La tarea debe tener al menos 10 caracteres")
        
        st.markdown("---")
        
        st.subheader("Técnicas a evaluar")
        zero_shot = st.checkbox("Zero-Shot", value=True)
        chain_of_thought = st.checkbox("Chain-of-Thought", value=True)
        react = st.checkbox("ReAct", value=True)
        self_consistency = st.checkbox("Self-Consistency", value=True)
        tree_of_thoughts = st.checkbox("Tree-of-Thoughts", value=True)
        
        st.markdown("---")
        
        if st.button("🔄 Ejecutar Benchmarks", type="primary", use_container_width=True):
            techniques = _get_selected_techniques(
                zero_shot, chain_of_thought, react, self_consistency, tree_of_thoughts
            )
            _execute_benchmarks(api_provider, task_mode, custom_task, techniques)
    
    # Main content
    results = load_benchmark_results()
    
    if results is None:
        _show_initial_info(api_keys_configured)
    else:
        # Prepare data
        if 'benchmark_df' not in st.session_state:
            st.session_state.benchmark_df = prepare_dataframe(results)
        
        df = st.session_state.benchmark_df
        
        # Summary metrics
        _show_results_summary(df, results)
        st.markdown("---")
        
        # Charts
        st.subheader("📈 Visualizaciones")
        fig_time, fig_tokens, fig_avg = create_comparison_charts(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_time, width='stretch')
        with col2:
            st.plotly_chart(fig_tokens, width='stretch')
        
        st.plotly_chart(fig_avg, width='stretch')
        st.markdown("---")
        
        # Detailed table
        _show_results_table(df)
        st.markdown("---")
        
        # Task details with prompts
        _show_task_details(results, df)


if __name__ == "__main__":
    main()

