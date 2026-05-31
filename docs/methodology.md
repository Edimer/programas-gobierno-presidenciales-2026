# Metodología

Este proyecto compara programas de gobierno presidenciales de Colombia 2026 con análisis semántico reproducible.

## Enfoque

1. **Corpus comparable**: cada PDF se convierte a texto y se divide en fragmentos solapados de longitud similar.
2. **Embeddings semánticos**: los fragmentos se convierten en vectores para comparar cercanía textual entre propuestas. En publicación se usan artefactos precomputados, de modo que la app no necesita recalcular el análisis al cargar.
3. **Ejes programáticos**: el posicionamiento no se interpreta como una similitud abstracta. Se calcula con una rúbrica explícita de ejes con dos polos sustantivos, por ejemplo mercado/Estado, punición/prevención o extractivismo/transición ecológica.
4. **Lectura visual**: el color ámbar indica inclinación hacia el polo de la izquierda del eje y el verde hacia el polo de la derecha. La etiqueta de cada celda resume la dirección y la intensidad como leve, media o fuerte.
5. **Cobertura de evidencia**: cada eje muestra qué proporción del programa contiene señales de la rúbrica. Un puntaje con baja cobertura debe leerse con más cautela.
6. **Agrupación**: la app usa un mapa semántico para ubicar fragmentos cercanos y detectar agrupaciones de propuestas.
7. **Conceptos distintivos**: el comparador prioriza frases de dos a cuatro palabras que aparecen con más fuerza en una candidatura que en las demás. Se excluyen nombres propios de candidaturas, números, pronombres, deícticos y expresiones genéricas.
8. **Afinidad ciudadana**: las respuestas de selección múltiple se comparan contra los fragmentos de cada candidatura con un vectorizador local. El puntaje final combina los fragmentos más cercanos por tema y pondera la importancia declarada de 1 a 10.

Repositorio del proyecto: [Edimer/programas-gobierno-presidenciales-2026](https://github.com/Edimer/programas-gobierno-presidenciales-2026).

## Precauciones

- Esto mide cercanía semántica textual, no verdad, viabilidad, calidad técnica ni conveniencia electoral.
- Los programas tienen extensiones diferentes; por eso se compara por fragmentos y no por documento completo.
- Los PDFs con texto mal extraído deben revisarse manualmente u OCR antes de publicar resultados definitivos.
- La visualización 2D simplifica un espacio de cientos o miles de dimensiones; debe leerse como mapa exploratorio.
- Los ejes programáticos son una lectura asistida por reglas transparentes; no reemplazan una codificación experta completa de ciencia política.
