# SfM Object Pipeline

Pipeline reproducible para reconstruccion de objetos con Structure from Motion usando COLMAP como backend principal.

## Docker

El proyecto puede ejecutarse dentro de Docker usando la imagen oficial `colmap/colmap` como base para evitar instalar COLMAP localmente.

### Construir la imagen del pipeline

```bash
docker compose build sfm
```

### Ejecutar con GPU NVIDIA

Si tienes NVIDIA Container Toolkit configurado, usa los servicios `sfm-gpu` y `colmap-gpu`.

```bash
docker compose build sfm
docker compose run --rm sfm-gpu prepare --object botella --config config.yaml
docker compose run --rm sfm-gpu extract-match --object botella --config config.yaml
docker compose run --rm sfm-gpu reconstruct --object botella --config config.yaml
docker compose run --rm sfm-gpu clean-model --object botella --config config.yaml
```

En `config.yaml` activa tambien:

```yaml
colmap:
  use_gpu: true
  gpu_index: 0
```

### Perfil recomendado para A100

Para una A100 usa [config.a100.yaml](/home/jd112003/Documents/USFQ/Semestre9/Computer_Vision/SFM/config.a100.yaml). Mantiene `sequential_matcher`, pero sube calidad y solapamiento:

- `max_image_size: 3200`
- `sift_max_num_features: 8192`
- `sequential_overlap: 20`
- `runs_per_size: 3`

Este perfil es el apropiado para procesar `leon` en hardware de servidor sin las restricciones de VRAM de la GTX 1660 Ti.

Ademas, el flujo recomendado incluye `clean-model` justo despues de `reconstruct` para producir una vista mas limpia del objeto en:

- `outputs/<objeto>/reconstruction/filtered_text_model/`

Para `detect-doppelgangers`, el comportamiento por defecto usa una base auxiliar separada con `exhaustive_matcher`. Esto permite buscar pares visualmente similares fuera del vecindario temporal del `sequential_matcher` de la reconstruccion principal, sin modificar `outputs/<objeto>/database.db`.

### Ejecutar el pipeline dentro del contenedor

```bash
docker compose run --rm sfm prepare --object botella --config config.yaml
docker compose run --rm sfm extract-match --object botella --config config.yaml
docker compose run --rm sfm reconstruct --object botella --config config.yaml
docker compose run --rm sfm clean-model --object botella --config config.yaml
docker compose run --rm sfm analyze-graph --object botella --config config.yaml
docker compose run --rm sfm detect-doppelgangers --object botella --config config.yaml
docker compose run --rm sfm run-ablation --object botella --config config.yaml
docker compose run --rm sfm report --object botella --config config.yaml
```

### Ejecutar COLMAP oficial directamente

```bash
docker compose run --rm colmap help
docker compose run --rm colmap feature_extractor -h
docker compose run --rm colmap-gpu feature_extractor -h
```

El `compose.yaml` monta el proyecto completo en `/workspace`, por lo que `data/`, `outputs/` y `config.yaml` se leen y escriben sobre tu carpeta local.

## Visor SSR de modelos filtrados

Hay un frontend en [frontend](/home/dchicaiza/SFM/frontend) hecho con Nuxt 3 + TypeScript para visualizar los modelos reconstruidos filtrados sin renderizar la nube completa.

Solo consume:

- `outputs/<objeto>/reconstruction/filtered_text_model/points3D.txt`

El servidor reduce el payload antes de enviarlo al navegador para no sobrecargar memoria y GPU del cliente.

### Levantar el visor con Docker Compose

```bash
docker compose build viewer
docker compose up -d viewer
```

Luego revisa el puerto publicado con:

```bash
docker ps | grep sfm-viewer
```

Y abre la URL correspondiente del host.

## Correr `leon` en una A100

### 1. Copiar el proyecto a la maquina remota

Debes llevar al menos:

- codigo del proyecto
- `compose.yaml`
- `Dockerfile`
- `config.a100.yaml`
- la carpeta `data/leon/`

### 2. Verificar GPU dentro de Docker

```bash
docker compose run --rm --entrypoint bash colmap-gpu -lc 'nvidia-smi'
```

### 3. Construir la imagen del pipeline

```bash
docker compose build sfm
```

### 4. Ejecutar `leon`

```bash
docker compose run --rm sfm-gpu prepare --object leon --config config.a100.yaml
docker compose run --rm sfm-gpu extract-match --object leon --config config.a100.yaml
docker compose run --rm sfm-gpu reconstruct --object leon --config config.a100.yaml
docker compose run --rm sfm-gpu clean-model --object leon --config config.a100.yaml
docker compose run --rm sfm-gpu analyze-graph --object leon --config config.a100.yaml
docker compose run --rm sfm-gpu detect-doppelgangers --object leon --config config.a100.yaml
docker compose run --rm sfm-gpu run-ablation --object leon --config config.a100.yaml
docker compose run --rm sfm-gpu report --object leon --config config.a100.yaml
```

O en una sola llamada:

```bash
bash ops/scripts/run_leon_a100.sh
```

### 5. Resultados

Los artefactos quedan en:

- `outputs/leon/reconstruction/`
- `outputs/leon/reconstruction/filtered_text_model/`
- `outputs/leon/graphs/`
- `outputs/leon/doppelgangers/`
- `outputs/leon/metrics/`

En `outputs/leon/doppelgangers/` el pipeline deja por defecto:

- `database.db`: base auxiliar para la busqueda exhaustiva de candidatos
- `doppelgangers.csv`
- `doppelgangers.json`
- `gallery.html`

## Estructura esperada

```text
data/
  objeto_a/
    images/
    metadata.yaml
outputs/
ops/
  scripts/
  logs/
config.yaml
config.a100.yaml
config.example.yaml
```

## Comandos principales

```bash
sfm-pipeline prepare --object botella --config config.yaml
sfm-pipeline extract-match --object botella --config config.yaml
sfm-pipeline reconstruct --object botella --config config.yaml
sfm-pipeline clean-model --object botella --config config.yaml
sfm-pipeline analyze-graph --object botella --config config.yaml
sfm-pipeline detect-doppelgangers --object botella --config config.yaml
sfm-pipeline run-ablation --object botella --config config.yaml
sfm-pipeline report --object botella --config config.yaml
```

## Notas

- Si usas Docker, no necesitas instalar `COLMAP` localmente.
- Para usar GPU en Docker necesitas NVIDIA Container Toolkit funcionando en el host.
- La GTX 1660 Ti corresponde a `gpu_index: 0` en una maquina con una sola GPU.
- Si ejecutas el pipeline fuera de Docker, `COLMAP` debe estar instalado y visible en `PATH`.
- Las graficas se exportan en `SVG` y el grafo de imagenes tambien en `DOT` y `HTML`.
- El pipeline evita dependencias pesadas para que las etapas de analisis y reporte funcionen incluso sin un entorno grafico.
- `clean-model` genera `outputs/<objeto>/reconstruction/filtered_text_model/` aplicando un filtrado no destructivo del modelo textual:
  - elimina puntos con poca evidencia geometrica
  - intenta remover un plano dominante con RANSAC
  - conserva el componente 3D mas compatible con un objeto centrado por las camaras
- Los scripts en `ops/scripts/` ya ejecutan `clean-model` por defecto despues de `reconstruct`.
- `detect-doppelgangers` usa por defecto `matcher: exhaustive` sobre una base auxiliar `outputs/<objeto>/doppelgangers/database.db`, de modo que no toca la base principal de reconstruccion.
- El umbral por defecto de doppelgangers se relajo a `max_inlier_ratio: 0.85` para capturar candidatos utiles en escenas reales donde los pares sospechosos no caen cerca de `0.25`.
