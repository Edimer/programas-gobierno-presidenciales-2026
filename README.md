# programas-gobierno-presidenciales-2026

- Los PDFs de cada programa fueron tomados desde las páginas oficiales y se pueden encontrar todos en este documento: https://ieu.unal.edu.co/wp-content/uploads/2026/05/elecciones-2026-2030-infografias.pdf

Explorador semantico de programas presidenciales de Colombia 2026. La app extrae texto de los PDFs, divide cada programa en fragmentos comparables, calcula embeddings multilingues y muestra:

- mapa semantico de propuestas,
- matriz de similitud entre candidaturas,
- terminos distintivos por programa,
- cuestionario ciudadano de hasta 10 preguntas con afinidad semantica.

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

Si el modelo de Hugging Face no descarga, el pipeline usa un respaldo TF-IDF para permitir exploracion preliminar. Para publicar, usar el modelo recomendado `intfloat/multilingual-e5-base` y revisar los diagnosticos de extraccion de texto.

## Fuentes y contexto

- Monitor electoral y tarjeton: [VotaBien](https://www.votabien.co/) y [AS/COA Poll Tracker](https://www.as-coa.org/articles/poll-tracker-colombias-2026-presidential-election).
- Contexto de encuestas recientes: [El Pais, mayo 2026](https://elpais.com/america-colombia/elecciones-presidenciales/2026-05-24/la-foto-final-de-las-encuestas-ivan-cepeda-primero-y-abelardo-de-la-espriella-se-distancia-de-paloma-valencia.html).
- Referencia tecnica: [Sentence Transformers STS](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html) y [Hugging Face multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base).
