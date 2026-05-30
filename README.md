# Programas presidenciales Colombia 2026

Explorador semántico de programas presidenciales de Colombia 2026. La app extrae texto de los PDFs, divide cada programa en fragmentos comparables, calcula embeddings de OpenAI y muestra:

- mapa semántico de propuestas,
- matriz de similitud relativa entre candidaturas,
- énfasis temáticos por candidatura,
- términos distintivos por programa,
- cuestionario ciudadano de selección múltiple con afinidad semántica.

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

## Fuentes

- Programa de Gobierno de Iván Cepeda: https://www.movimientopactohistorico.co/programa-gobierno
- Programa de Gobierno de Abelardo de la Espriella: https://defensoresdelapatria.com/wpcontent/uploads/2026/04/PROPUESTAS-DEL-TIGRE.pdf
- Programa de Gobierno de Paloma Valencia: https://palomavalencia.com/images/documentos/Plan%20Integrado%20de%20Gobierno%20Final_compressed.pdf
- Programa de Gobierno de Claudia López: https://claudia-lopez.com/campana/
- Programa de Gobierno de Sergio Fajardo: https://drive.google.com/file/d/1b0JOU1qalqmih9YdYbuWIDWQEAAvYWn7/view

## Notas metodológicas

Esto mide cercanía semántica textual, no verdad, viabilidad, calidad técnica ni conveniencia electoral. Los programas tienen extensiones y formatos distintos; por eso la app compara fragmentos y muestra diagnósticos de extracción.
