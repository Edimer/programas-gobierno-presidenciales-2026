# Metodologia

Este proyecto compara programas de gobierno presidenciales de Colombia 2026 con analisis semantico reproducible.

## Enfoque

1. **Corpus comparable**: cada PDF se convierte a texto y se divide en fragmentos solapados de longitud similar.
2. **Embeddings**: por defecto se usa `intfloat/multilingual-e5-base`, un modelo multilingue disponible en Hugging Face y compatible con `sentence-transformers`. Los programas se codifican con prefijo `passage:` y las respuestas ciudadanas con `query:`, siguiendo la recomendacion del model card.
3. **Similitud**: se calcula coseno entre embeddings normalizados.
4. **Agrupacion**: la app usa UMAP para proyeccion 2D y HDBSCAN para clusters. Si esas librerias no estan disponibles, usa PCA y KMeans como respaldo.
5. **Afinidad ciudadana**: las respuestas abiertas se embeben y se comparan contra los fragmentos de cada candidatura. El puntaje final combina los fragmentos mas similares por tema y pondera por importancia declarada.

## Precauciones

- Esto mide cercania semantica textual, no verdad, viabilidad, calidad tecnica ni conveniencia electoral.
- Los programas tienen extensiones diferentes; por eso se compara por fragmentos y no por documento completo.
- Los PDFs con texto mal extraido deben revisarse manualmente u OCR antes de publicar resultados definitivos.
- La visualizacion 2D simplifica un espacio de cientos de dimensiones; debe leerse como mapa exploratorio.
