# IfachGuard

## Dominio

Asistente conversacional centrado en el Parque Natural del Penyal d'Ifac (Calpe). Responde consultas sobre normativa, uso público, flora y fauna y vías de escalada, utilizando como base el corpus del proyecto: legislación oficial vigente del parque y documentación complementaria.

## System prompt

El prompt define a IfachGuard como asistente especializado y delimita los temas que puede atender. Indica que las respuestas sobre el parque deben apoyarse en los fragmentos recuperados, citar su procedencia y reconocer cuando la evidencia no basta. También distingue normativa de guías complementarias, incorpora criterios para interpretar las modificaciones del PRUG y establece guardrails: atiende preguntas informativas sobre actividades reguladas, pero no ofrece instrucciones para eludir controles o infringir normas. Estas decisiones buscan que las respuestas sean pertinentes, trazables al corpus y prudentes ante instrucciones maliciosas o fuentes de distinta naturaleza.

## Requisitos

- Python y pip.
- Dependencias de `requirements.txt` (incluye Streamlit, LangChain, LangGraph, ChromaDB y la integración con Google Gemini).
- Una API key de Gemini con acceso a los modelos configurados.

El notebook lee la clave desde `.env` en la raíz del proyecto:

```text
GEMINI_API_KEY=tu_clave_de_google_gemini
```

Para Streamlit local, copia `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml` y añade la misma clave:

```powershell
Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
GEMINI_API_KEY = "tu_clave_de_google_gemini"
```

No subas `.env` ni `.streamlit/secrets.toml` a GitHub; ambos están excluidos por `.gitignore`.

## Instalación y notebook

Clona el repositorio, crea un entorno virtual e instala las dependencias:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Abre `IfachGuard.ipynb` en VS Code o Jupyter y ejecuta las celdas en orden. El notebook carga la clave desde `.env` y utiliza el corpus de `data/` y el índice persistido en `chroma_db_ifach/`.

## Aplicación Streamlit

Para ejecutarla localmente, activa el entorno virtual, configura `.streamlit/secrets.toml` y ejecuta:

```powershell
streamlit run streamlit_app.py
```

Para desplegarla en Streamlit Community Cloud:

1. Sube a GitHub `streamlit_app.py`, `ifachguard_agent.py`, `requirements.txt`, `IfachGuard.ipynb`, `data/` y `chroma_db_ifach/`.
2. Crea una aplicación en Streamlit Community Cloud desde el repositorio y selecciona `streamlit_app.py` como archivo principal.
3. En **Settings > Secrets**, configura:

```toml
GEMINI_API_KEY = "tu_clave_de_google_gemini"
```

El enunciado del proyecto y los secretos locales están excluidos del repositorio.
