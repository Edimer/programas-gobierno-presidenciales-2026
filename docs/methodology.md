# Metodología

Este proyecto compara programas de gobierno presidenciales de Colombia 2026 con análisis semántico reproducible.

## Enfoque

1. **Corpus comparable**: cada PDF se convierte a texto y se divide en fragmentos solapados de longitud similar.
2. **Embeddings**: por defecto se usa `text-embedding-3-large` de OpenAI. La clave se lee desde `API_OPENAI` u `OPENAI_API_KEY` en `.env`.
3. **Similitud relativa**: los vectores se normalizan y, para la matriz entre candidaturas, se centran frente al corpus completo antes de calcular centroides. Esto evita que todos los programas aparezcan con similitudes cercanas a 1.00 solo por compartir vocabulario político general.
4. **Agrupación**: la app usa un mapa semántico para ubicar fragmentos cercanos y detectar agrupaciones de propuestas.
5. **Conceptos distintivos**: el comparador prioriza frases de dos a cuatro palabras que aparecen con más fuerza en una candidatura que en las demás. Se excluyen nombres propios de candidaturas, números, pronombres, deícticos y expresiones genéricas.
6. **Afinidad ciudadana**: las respuestas de selección múltiple se embeben y se comparan contra los fragmentos de cada candidatura. El puntaje final combina los fragmentos más similares por tema y pondera la importancia declarada de 1 a 10.

Repositorio del proyecto: [Edimer/programas-gobierno-presidenciales-2026](https://github.com/Edimer/programas-gobierno-presidenciales-2026).

## Precauciones

- Esto mide cercanía semántica textual, no verdad, viabilidad, calidad técnica ni conveniencia electoral.
- Los programas tienen extensiones diferentes; por eso se compara por fragmentos y no por documento completo.
- Los PDFs con texto mal extraído deben revisarse manualmente u OCR antes de publicar resultados definitivos.
- La visualización 2D simplifica un espacio de cientos o miles de dimensiones; debe leerse como mapa exploratorio.
