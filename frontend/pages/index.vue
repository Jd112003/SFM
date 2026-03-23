<script setup lang="ts">
import PointCloudViewer from '~/components/PointCloudViewer.client.vue'
import type { ModelPayload, ModelSummary } from '~/types/models'

type ColorMode = 'rgb' | 'height' | 'mono'
type BackgroundMode = 'light' | 'dark' | 'sand'
type NavigationMode = 'orbit' | 'map'
type ProjectionMode = 'perspective' | 'orthographic'
type VisualizationMode = 'model' | 'cameras' | 'both'

const viewerRef = ref<InstanceType<typeof PointCloudViewer> | null>(null)
const DEFAULT_MODEL_ID = 'leon'
const MAX_POINT_SIZE = 0.05

const pointBudget = ref<number>(25000)
const pointSize = ref<number>(MAX_POINT_SIZE)
const showAxes = ref<boolean>(true)
const showGrid = ref<boolean>(false)
const colorMode = ref<ColorMode>('rgb')
const backgroundMode = ref<BackgroundMode>('light')
const navigationMode = ref<NavigationMode>('orbit')
const projectionMode = ref<ProjectionMode>('perspective')
const visualizationMode = ref<VisualizationMode>('model')
const cameraSize = ref<number>(0.18)

const { data: models, pending: modelsPending, error: modelsError } = await useFetch<ModelSummary[]>('/api/models')
const initialSelectedId = models.value?.find((model) => model.id === DEFAULT_MODEL_ID)?.id ?? models.value?.[0]?.id ?? ''
const selectedId = ref<string>(initialSelectedId)

watchEffect(() => {
  if (!models.value?.length) {
    return
  }

  const defaultModel = models.value.find((model) => model.id === DEFAULT_MODEL_ID)

  if (!selectedId.value || !models.value.some((model) => model.id === selectedId.value)) {
    selectedId.value = defaultModel?.id ?? models.value[0].id
  }
})

const selectedSummary = computed(() => models.value?.find((model) => model.id === selectedId.value) ?? null)

const {
  data: modelPayload,
  pending: modelPending,
  refresh: refreshModel
} = await useFetch<ModelPayload>(
  () => (selectedId.value ? `/api/models/${selectedId.value}` : '/api/models'),
  {
    query: computed(() => ({ maxPoints: pointBudget.value })),
    immediate: false,
    watch: false
  }
)

watch([selectedId, pointBudget], async () => {
  if (selectedId.value) {
    await refreshModel()
  }
}, { immediate: true })

const reductionLabel = computed(() => {
  if (!selectedSummary.value || !modelPayload.value) {
    return null
  }
  const ratio = (modelPayload.value.points / selectedSummary.value.originalPoints) * 100
  return `${modelPayload.value.points.toLocaleString()} / ${selectedSummary.value.originalPoints.toLocaleString()}`
    + ` (${ratio.toFixed(1)}%)`
})
</script>

<template>
  <main class="page">
    <section class="layout">
      <aside class="panel controls">
        <div class="block">
          <h2>Modelos</h2>
          <p v-if="modelsPending" class="muted">Cargando…</p>
          <p v-else-if="modelsError" class="muted">No se pudieron cargar los modelos.</p>
          <template v-else>
            <label class="field">
              <span>Modelo</span>
              <select v-model="selectedId">
                <option v-for="model in models" :key="model.id" :value="model.id">
                  {{ model.name }}
                </option>
              </select>
            </label>
            <label class="field">
              <span>Puntos</span>
              <select v-model.number="pointBudget">
                <option :value="10000">10,000</option>
                <option :value="25000">25,000</option>
                <option :value="40000">40,000</option>
                <option :value="60000">60,000</option>
              </select>
            </label>
          </template>
        </div>

        <div class="block">
          <h2>Vista</h2>
          <label class="field">
            <span>Navegacion</span>
            <select v-model="navigationMode">
              <option value="map">Map (arrastrar para desplazarte)</option>
              <option value="orbit">Orbit (orbitar alrededor del modelo)</option>
            </select>
          </label>
          <label class="field">
            <span>Camara</span>
            <select v-model="projectionMode">
              <option value="perspective">Perspective</option>
              <option value="orthographic">Orthographic</option>
            </select>
          </label>
          <label class="field">
            <span>Visualizacion</span>
            <select v-model="visualizationMode">
              <option value="model">Modelo</option>
              <option value="cameras">Camaras</option>
              <option value="both">Modelo + camaras</option>
            </select>
          </label>
          <p class="hint" v-if="navigationMode === 'map'">
            Modo Map: click izquierdo para pan, click derecho para rotar, rueda para zoom al cursor.
          </p>
          <p class="hint" v-else>
            Modo Orbit: click izquierdo rota, click derecho desplaza, rueda para zoom al cursor.
          </p>
          <div class="button-grid">
            <button type="button" @click="viewerRef?.fitView()">Fit</button>
            <button type="button" @click="viewerRef?.recenterView()">Center</button>
            <button type="button" @click="viewerRef?.resetView()">Iso</button>
            <button type="button" @click="viewerRef?.setFrontView()">Front</button>
            <button type="button" @click="viewerRef?.setSideView()">Side</button>
            <button type="button" @click="viewerRef?.setTopView()">Top</button>
          </div>
        </div>

        <div class="block">
          <h2>Render</h2>
          <label class="field">
            <span>Color</span>
            <select v-model="colorMode">
              <option value="rgb">RGB</option>
              <option value="height">Height</option>
              <option value="mono">Mono</option>
            </select>
          </label>
          <label class="field">
            <span>Fondo</span>
            <select v-model="backgroundMode">
              <option value="light">Light</option>
              <option value="sand">Sand</option>
              <option value="dark">Dark</option>
            </select>
          </label>
          <label class="field">
            <span>Tamano de punto: {{ pointSize.toFixed(3) }}</span>
            <input v-model.number="pointSize" type="range" min="0.006" max="0.05" step="0.002">
          </label>
          <label class="field">
            <span>Tamano de camaras: {{ cameraSize.toFixed(3) }}</span>
            <input v-model.number="cameraSize" type="range" min="0.05" max="0.5" step="0.01">
          </label>
          <label class="toggle">
            <input v-model="showAxes" type="checkbox">
            <span>Ejes</span>
          </label>
          <label class="toggle">
            <input v-model="showGrid" type="checkbox">
            <span>Grid</span>
          </label>
        </div>

        <div v-if="selectedSummary" class="block stats">
          <div class="stat">
            <strong>Original</strong>
            <span>{{ selectedSummary.originalPoints.toLocaleString() }}</span>
          </div>
          <div class="stat">
            <strong>Filtrado</strong>
            <span>{{ selectedSummary.points.toLocaleString() }}</span>
          </div>
          <div v-if="reductionLabel" class="stat">
            <strong>Payload</strong>
            <span>{{ reductionLabel }}</span>
          </div>
        </div>
      </aside>

      <section class="panel viewer-panel">
        <div class="viewer-head">
          <h1>{{ selectedSummary?.name ?? 'Modelo' }}</h1>
        </div>

        <div v-if="modelPending" class="placeholder">
          Renderizando…
        </div>
        <PointCloudViewer
          v-else-if="modelPayload"
          ref="viewerRef"
          :model="modelPayload"
          :point-size="pointSize"
          :show-axes="showAxes"
          :show-grid="showGrid"
          :color-mode="colorMode"
          :background-mode="backgroundMode"
          :navigation-mode="navigationMode"
          :projection-mode="projectionMode"
          :visualization-mode="visualizationMode"
          :camera-size="cameraSize"
        />
      </section>
    </section>
  </main>
</template>

<style scoped>
.page {
  padding: 20px;
}

.layout {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 20px;
}

.panel {
  border: 1px solid var(--line);
  border-radius: 26px;
  background: var(--panel);
  backdrop-filter: blur(14px);
  box-shadow: 0 18px 40px rgba(77, 58, 36, 0.08);
}

.controls,
.viewer-panel {
  padding: 18px;
}

.block + .block {
  margin-top: 18px;
  padding-top: 18px;
  border-top: 1px solid rgba(160, 140, 118, 0.2);
}

h1,
h2 {
  margin: 0 0 12px;
  font-family: "Iowan Old Style", Georgia, serif;
}

h1 {
  font-size: 1.7rem;
}

h2 {
  font-size: 1.06rem;
}

.field {
  display: grid;
  gap: 8px;
  margin-bottom: 14px;
}

.field span,
.muted {
  color: var(--muted);
  font-size: 0.92rem;
}

.hint {
  margin: -4px 0 12px;
  color: var(--muted);
  font-size: 0.85rem;
  line-height: 1.35;
}

.field select,
.field input[type='range'],
.button-grid button {
  width: 100%;
}

.field select {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(255, 255, 255, 0.9);
}

.button-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.button-grid button {
  border: 1px solid rgba(165, 83, 45, 0.18);
  background: rgba(165, 83, 45, 0.08);
  color: var(--accent);
  padding: 11px 12px;
  border-radius: 12px;
  cursor: pointer;
}

.toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
  color: var(--muted);
}

.stats {
  display: grid;
  gap: 10px;
}

.stat {
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(159, 140, 118, 0.18);
  background: rgba(255, 255, 255, 0.72);
}

.stat strong {
  display: block;
  margin-bottom: 4px;
}

.viewer-head {
  margin-bottom: 14px;
}

.placeholder {
  display: grid;
  place-items: center;
  min-height: 68vh;
  border-radius: 24px;
  border: 1px dashed var(--line);
  color: var(--muted);
}

@media (max-width: 980px) {
  .page {
    padding: 14px;
  }

  .layout {
    grid-template-columns: 1fr;
  }
}
</style>
