# 🚀 Guía de Despliegue - Prompt Engineering Toolkit

Esta guía te ayudará a desplegar tu aplicación Streamlit en diferentes plataformas.

## 📋 Opciones de Despliegue

### 1. Streamlit Cloud (Recomendado - Gratis) ⭐

Streamlit Cloud es la forma más fácil y gratuita de desplegar tu aplicación.

#### Pasos para desplegar:

1. **Sube tu código a GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit dashboard"
   git push origin main
   ```

2. **Ve a Streamlit Cloud**
   - Visita: https://streamlit.io/cloud
   - Inicia sesión con tu cuenta de GitHub

3. **Conecta tu repositorio**
   - Haz clic en "New app"
   - Selecciona tu repositorio
   - Selecciona la rama (main)
   - Ruta del archivo: `streamlit_app.py`

4. **Configura variables de entorno**
   - En la configuración de la app, agrega:
     - `OPENAI_API_KEY`: Tu API key de OpenAI
     - `ANTHROPIC_API_KEY`: Tu API key de Anthropic (opcional)

5. **Despliega**
   - Haz clic en "Deploy"
   - Tu app estará disponible en: `https://tu-app.streamlit.app`

#### ⚠️ Nota sobre API Keys:
Para seguridad, **NO** subas tu archivo `.env` a GitHub. En su lugar:
- Usa las variables de entorno de Streamlit Cloud
- O usa Streamlit Secrets para manejar credenciales

### 2. Heroku

#### Requisitos previos:
```bash
pip install gunicorn
```

#### Archivos necesarios:

**Procfile** (ya creado):
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

**setup.sh** (ya creado):
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
```

#### Pasos:

1. **Instala Heroku CLI**: https://devcenter.heroku.com/articles/heroku-cli

2. **Login y crea app**:
   ```bash
   heroku login
   heroku create tu-app-nombre
   ```

3. **Configura variables de entorno**:
   ```bash
   heroku config:set OPENAI_API_KEY=tu_api_key
   heroku config:set ANTHROPIC_API_KEY=tu_api_key
   ```

4. **Despliega**:
   ```bash
   git push heroku main
   ```

### 3. Docker

#### Dockerfile (ya creado):
El Dockerfile está configurado para ejecutar la aplicación.

#### Pasos:

1. **Construye la imagen**:
   ```bash
   docker build -t prompt-engineering-app .
   ```

2. **Ejecuta el contenedor**:
   ```bash
   docker run -p 8501:8501 \
     -e OPENAI_API_KEY=tu_api_key \
     -e ANTHROPIC_API_KEY=tu_api_key \
     prompt-engineering-app
   ```

3. **Despliega en servicios cloud**:
   - **Google Cloud Run**: `gcloud run deploy`
   - **AWS ECS/Fargate**: Usa el Dockerfile
   - **Azure Container Instances**: Usa el Dockerfile
   - **DigitalOcean App Platform**: Conecta tu repositorio

### 4. VPS (Servidor Virtual Privado)

#### Opciones populares:
- **DigitalOcean Droplet**
- **AWS EC2**
- **Google Cloud Compute Engine**
- **Linode**

#### Pasos generales:

1. **Conecta a tu servidor**:
   ```bash
   ssh usuario@tu-servidor-ip
   ```

2. **Instala dependencias**:
   ```bash
   sudo apt update
   sudo apt install python3-pip
   pip3 install -r requirements.txt
   ```

3. **Ejecuta con nohup o systemd**:
   ```bash
   nohup streamlit run streamlit_app.py --server.port 8501 &
   ```

4. **O usa un proceso manager** (recomendado):
   - **PM2**: `pm2 start streamlit_app.py`
   - **systemd**: Crea un servicio systemd

### 5. Railway

Railway es otra opción fácil y gratuita.

#### Pasos:

1. **Conecta tu repositorio** en railway.app
2. **Configura variables de entorno**
3. **Railway detectará automáticamente** que es una app Streamlit

## 🔒 Seguridad - Variables de Entorno

### ⚠️ IMPORTANTE: Nunca subas API keys a GitHub

1. **Agrega `.env` a `.gitignore`** (ya está incluido)
2. **Usa variables de entorno** en la plataforma de despliegue
3. **Para Streamlit Cloud**, usa Secrets:
   - Crea un archivo `.streamlit/secrets.toml` localmente (NO lo subas)
   - En Streamlit Cloud, ve a Settings > Secrets y agrega:
     ```toml
     OPENAI_API_KEY = "tu_api_key"
     ANTHROPIC_API_KEY = "tu_api_key"
     ```

## 📝 Checklist Pre-Despliegue

- [ ] Código subido a GitHub
- [ ] `.env` en `.gitignore`
- [ ] `requirements.txt` actualizado
- [ ] API keys configuradas como variables de entorno
- [ ] Probado localmente
- [ ] README actualizado con link de despliegue

## 🎯 Recomendación

**Para proyectos de portfolio/demostración:**
- ✅ **Streamlit Cloud** - Más fácil, gratis, perfecto para mostrar tu trabajo

**Para producción:**
- ✅ **Docker + Cloud Run/AWS/Azure** - Más control y escalabilidad

## 📚 Recursos

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy)
- [Heroku Python Guide](https://devcenter.heroku.com/articles/getting-started-with-python)

## 🆘 Troubleshooting

### Error: "Module not found"
- Verifica que `requirements.txt` incluya todas las dependencias
- Asegúrate de que el archivo esté en la raíz del proyecto

### Error: "API Key not found"
- Verifica que las variables de entorno estén configuradas
- En Streamlit Cloud, revisa Settings > Secrets

### Error: "Port already in use"
- Cambia el puerto en la configuración
- O usa variables de entorno para el puerto

