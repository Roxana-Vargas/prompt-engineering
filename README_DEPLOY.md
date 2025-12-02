# 🚀 Despliegue Rápido - Streamlit Cloud

## Pasos para desplegar en Streamlit Cloud (5 minutos)

### 1. Prepara tu repositorio

```bash
# Asegúrate de que todo esté commitado
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 2. Ve a Streamlit Cloud

1. Visita: https://streamlit.io/cloud
2. Inicia sesión con GitHub
3. Haz clic en "New app"

### 3. Configura la app

- **Repository**: Selecciona tu repositorio
- **Branch**: `main`
- **Main file path**: `streamlit_app.py`

### 4. Configura Secrets (API Keys)

En la sección "Secrets", agrega:

```toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."  # Opcional
```

### 5. Despliega

Haz clic en "Deploy" y espera ~2 minutos.

### ✅ ¡Listo!

Tu app estará disponible en:
`https://tu-repo-name.streamlit.app`

## 🔒 Seguridad

- ✅ **NO** subas tu archivo `.env` a GitHub
- ✅ Usa Secrets de Streamlit Cloud para API keys
- ✅ El archivo `.env` ya está en `.gitignore`

## 📝 Checklist

- [ ] Código en GitHub
- [ ] `requirements.txt` actualizado
- [ ] `.env` NO está en el repositorio
- [ ] API keys configuradas en Secrets
- [ ] App desplegada y funcionando

## 🆘 Problemas Comunes

**Error: "Module not found"**
- Verifica que `requirements.txt` tenga todas las dependencias

**Error: "API Key not found"**
- Revisa que las Secrets estén configuradas correctamente
- Verifica los nombres: `OPENAI_API_KEY` (no `OPENAI_API_KEY` con espacios)

**App no carga**
- Revisa los logs en Streamlit Cloud
- Verifica que el archivo principal sea `streamlit_app.py`

