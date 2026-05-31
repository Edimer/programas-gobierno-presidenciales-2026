# Programas presidenciales Colombia 2026

Explorador semántico de programas presidenciales de Colombia 2026. La app extrae texto de los PDFs, divide cada programa en fragmentos comparables, calcula embeddings de OpenAI y muestra:

- mapa semántico de propuestas,
- ejes programáticos interpretables por candidatura,
- lectura temática normalizada dentro de cada programa,
- conceptos distintivos por programa,
- cuestionario ciudadano de selección múltiple con afinidad textual local.

## Uso local

En Windows, doble clic en:

```text
run_app.bat
```

El archivo crea el entorno virtual si hace falta, instala dependencias, genera los artefactos iniciales y abre la app.

Uso manual:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/build_artifacts.py
streamlit run app.py
```

La app espera una variable `API_OPENAI` u `OPENAI_API_KEY` en `.env`. También puede definirse `OPENAI_EMBEDDING_MODEL`; si no se define, se usa `text-embedding-3-large`.

## Despliegue en Streamlit Cloud

La app está preparada para publicarse con artefactos precomputados. Esto evita recalcular embeddings al cargar y reduce el riesgo de errores por límites de API.

Antes de hacer `push`, verifique que estén incluidos estos elementos:

- `data/artifacts/` con los archivos `.parquet`, `.npy` y `diagnostics_*.json`.
- `requirements.txt`.
- `.streamlit/secrets.toml` no debe subirse al repositorio; los secretos se configuran en Streamlit Cloud.
- `.env` no debe subirse al repositorio.

Si necesita regenerar los artefactos localmente:

```powershell
.\.venv\Scripts\python.exe scripts\build_artifacts.py
```

En Streamlit Cloud, configure la variable `API_OPENAI` u `OPENAI_API_KEY` solo si va a regenerar o ampliar análisis que requieran la API. Para la versión publicada con artefactos precomputados, la app no debería llamar a OpenAI durante la carga inicial ni al calcular la afinidad ciudadana.

## Fuentes

- Programa de Gobierno de Iván Cepeda: https://www.movimientopactohistorico.co/programa-gobierno
- Programa de Gobierno de Abelardo de la Espriella: https://defensoresdelapatria.com/wpcontent/uploads/2026/04/PROPUESTAS-DEL-TIGRE.pdf
- Programa de Gobierno de Paloma Valencia: https://palomavalencia.com/images/documentos/Plan%20Integrado%20de%20Gobierno%20Final_compressed.pdf
- Programa de Gobierno de Claudia López: https://claudia-lopez.com/campana/
- Programa de Gobierno de Sergio Fajardo: https://drive.google.com/file/d/1b0JOU1qalqmih9YdYbuWIDWQEAAvYWn7/view

## Notas metodológicas

Esto mide cercanía semántica textual, no verdad, viabilidad, calidad técnica ni conveniencia electoral. Los programas tienen extensiones y formatos distintos; por eso la app compara fragmentos, muestra diagnósticos de extracción y evita comparar candidaturas por volumen de páginas.

Los ejes programáticos son una lectura asistida por reglas explícitas. Cada eje tiene dos polos sustantivos, por ejemplo mercado/Estado o punición/prevención, y la visualización muestra dirección e intensidad de forma exploratoria.
