<script setup lang="ts">
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { ModelPayload } from '~/types/models'

type CameraPreset = 'iso' | 'front' | 'side' | 'top'
type ColorMode = 'rgb' | 'height' | 'mono'
type BackgroundMode = 'light' | 'dark' | 'sand'
type NavigationMode = 'orbit' | 'map'
type ProjectionMode = 'perspective' | 'orthographic'
type VisualizationMode = 'model' | 'cameras' | 'both'

const props = defineProps<{
  model: ModelPayload | null
  pointSize: number
  showAxes: boolean
  showGrid: boolean
  colorMode: ColorMode
  backgroundMode: BackgroundMode
  navigationMode: NavigationMode
  projectionMode: ProjectionMode
  visualizationMode: VisualizationMode
  cameraSize: number
}>()

const container = ref<HTMLDivElement | null>(null)

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | THREE.OrthographicCamera | null = null
let controls: OrbitControls | null = null
let pointCloud: THREE.Points | null = null
let cameraGroup: THREE.Group | null = null
let axesHelper: THREE.AxesHelper | null = null
let gridHelper: THREE.GridHelper | null = null
let currentModel: ModelPayload | null = null
let frameHandle = 0
let activePreset: CameraPreset = 'iso'

const raycaster = new THREE.Raycaster()
const pointer = new THREE.Vector2()

function cameraDistanceForModel(size: THREE.Vector3) {
  return Math.max(size.length() * 0.75, 1.6)
}

function boundsFromPositions(positions: number[]) {
  const min = new THREE.Vector3(Infinity, Infinity, Infinity)
  const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity)

  for (let i = 0; i < positions.length; i += 3) {
    min.x = Math.min(min.x, positions[i])
    min.y = Math.min(min.y, positions[i + 1])
    min.z = Math.min(min.z, positions[i + 2])
    max.x = Math.max(max.x, positions[i])
    max.y = Math.max(max.y, positions[i + 1])
    max.z = Math.max(max.z, positions[i + 2])
  }

  if (!Number.isFinite(min.x) || !Number.isFinite(max.x)) {
    return {
      min: new THREE.Vector3(-0.5, -0.5, -0.5),
      max: new THREE.Vector3(0.5, 0.5, 0.5)
    }
  }

  return { min, max }
}

function modelMetrics(model: ModelPayload) {
  const useModel = props.visualizationMode !== 'cameras'
  const useCameras = props.visualizationMode !== 'model'
  const datasets: number[][] = []

  if (useModel) {
    datasets.push(model.positions)
  }
  if (useCameras) {
    datasets.push(model.cameraPositions)
  }

  const min = new THREE.Vector3(Infinity, Infinity, Infinity)
  const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity)

  for (const dataset of datasets) {
    const bounds = boundsFromPositions(dataset)
    min.min(bounds.min)
    max.max(bounds.max)
  }

  const center = min.clone().add(max).multiplyScalar(0.5)
  const size = max.clone().sub(min)
  const radius = Math.max(size.length() * 0.5, 0.5)
  return { center, size, radius }
}

function backgroundColor(mode: BackgroundMode) {
  if (mode === 'dark') return '#101215'
  if (mode === 'sand') return '#ece3d4'
  return '#f6f1e7'
}

function cameraMaterialSize() {
  return Math.max(props.cameraSize, 0.05)
}

function disposeViewer() {
  if (frameHandle) {
    cancelAnimationFrame(frameHandle)
    frameHandle = 0
  }
  controls?.dispose()
  controls = null
  if (pointCloud && scene) {
    scene.remove(pointCloud)
    pointCloud.geometry.dispose()
    ;(pointCloud.material as THREE.Material).dispose()
  }
  pointCloud = null
  if (cameraGroup && scene) {
    scene.remove(cameraGroup)
    cameraGroup.traverse((child) => {
      if ('geometry' in child && child.geometry) {
        child.geometry.dispose()
      }
      if ('material' in child && child.material) {
        const material = child.material
        if (Array.isArray(material)) {
          material.forEach((entry) => entry.dispose())
        } else {
          material.dispose()
        }
      }
    })
  }
  cameraGroup = null
  renderer?.dispose()
  renderer = null
}

function updateCameraFrustum(center: THREE.Vector3, radius: number) {
  if (!camera || !container.value) {
    return
  }

  if (camera instanceof THREE.PerspectiveCamera) {
    camera.near = Math.max(radius / 500, 0.01)
    camera.far = Math.max(radius * 40, 250)
    camera.updateProjectionMatrix()
    return
  }

  const aspect = Math.max(container.value.clientWidth / Math.max(container.value.clientHeight, 1), 1e-6)
  const halfHeight = radius * 1.2
  const halfWidth = halfHeight * aspect
  camera.left = -halfWidth
  camera.right = halfWidth
  camera.top = halfHeight
  camera.bottom = -halfHeight
  camera.near = -radius * 20
  camera.far = radius * 40
  camera.position.sub(controls?.target ?? new THREE.Vector3()).add(center)
  camera.updateProjectionMatrix()
}

function applyCameraPreset(preset: CameraPreset) {
  if (!camera || !controls || !currentModel) {
    return
  }
  activePreset = preset
  const { center, size, radius } = modelMetrics(currentModel)
  const distance = cameraDistanceForModel(size)
  const offset = new THREE.Vector3(distance, distance * 0.8, distance)

  if (preset === 'front') {
    offset.set(0, 0, distance * 1.5)
  } else if (preset === 'side') {
    offset.set(distance * 1.5, 0, 0)
  } else if (preset === 'top') {
    offset.set(0, distance * 1.8, 0)
  }

  camera.position.copy(center).add(offset)
  controls.target.copy(center)
  controls.minDistance = Math.max(radius * 0.08, 0.08)
  controls.maxDistance = Math.max(radius * 12, 50)
  updateCameraFrustum(center, radius)
  controls.update()
}

function configureControls() {
  if (!camera || !renderer) {
    return
  }

  controls?.dispose()

  controls = new OrbitControls(camera, renderer.domElement)

  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.zoomToCursor = true
  controls.screenSpacePanning = true
  controls.panSpeed = props.navigationMode === 'map' ? 1.35 : 1
  controls.rotateSpeed = props.navigationMode === 'map' ? 0.72 : 0.92
  controls.zoomSpeed = props.navigationMode === 'map' ? 1.3 : 1.1
  controls.keyPanSpeed = props.navigationMode === 'map' ? 10 : 7
  controls.listenToKeyEvents(window)
  controls.enablePan = true
  controls.enableZoom = true
  controls.enableRotate = true

  // Emulate map-style controls with OrbitControls for compatibility across Three.js versions.
  if (props.navigationMode === 'map') {
    controls.mouseButtons.LEFT = THREE.MOUSE.PAN
    controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE
    controls.touches.ONE = THREE.TOUCH.PAN
    controls.touches.TWO = THREE.TOUCH.DOLLY_ROTATE
  } else {
    controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE
    controls.mouseButtons.RIGHT = THREE.MOUSE.PAN
    controls.touches.ONE = THREE.TOUCH.ROTATE
    controls.touches.TWO = THREE.TOUCH.DOLLY_PAN
  }

  if (currentModel) {
    const { radius } = modelMetrics(currentModel)
    controls.minDistance = Math.max(radius * 0.08, 0.08)
    controls.maxDistance = Math.max(radius * 12, 50)
  }

  controls.saveState()
}

function fitToModel() {
  if (!currentModel) {
    return
  }
  applyCameraPreset(activePreset)
}

function recenterOnTarget(nextTarget: THREE.Vector3) {
  if (!camera || !controls || !currentModel) {
    return
  }
  const offset = camera.position.clone().sub(controls.target)
  controls.target.copy(nextTarget)
  camera.position.copy(nextTarget).add(offset)
  updateCameraFrustum(nextTarget, modelMetrics(currentModel).radius)
  controls.update()
}

function handleDoubleClick(event: MouseEvent) {
  if (!renderer || !camera || !pointCloud || !container.value || props.visualizationMode === 'cameras') {
    return
  }

  const rect = renderer.domElement.getBoundingClientRect()
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
  raycaster.params.Points.threshold = Math.max(props.pointSize * 12, 0.08)
  raycaster.setFromCamera(pointer, camera)
  const hits = raycaster.intersectObject(pointCloud)

  if (hits.length > 0 && hits[0]?.point) {
    recenterOnTarget(hits[0].point.clone())
    return
  }

  fitToModel()
}

function buildDisplayColors(model: ModelPayload, mode: ColorMode) {
  if (mode === 'rgb') {
    return model.colors
  }
  if (mode === 'mono') {
    const mono: number[] = []
    for (let i = 0; i < model.positions.length; i += 3) {
      mono.push(0.18, 0.18, 0.18)
    }
    return mono
  }

  const colors: number[] = []
  const minY = model.bounds.min[1]
  const maxY = model.bounds.max[1]
  const span = Math.max(maxY - minY, 1e-6)

  for (let i = 0; i < model.positions.length; i += 3) {
    const y = model.positions[i + 1]
    const t = (y - minY) / span
    colors.push(0.18 + t * 0.72, 0.35 + (1 - Math.abs(t - 0.5) * 2) * 0.4, 0.82 - t * 0.52)
  }
  return colors
}

function applyVisualSettings() {
  if (!scene || !renderer) {
    return
  }
  scene.background = new THREE.Color(backgroundColor(props.backgroundMode))

  if (axesHelper) {
    axesHelper.visible = props.showAxes
  }
  if (gridHelper) {
    gridHelper.visible = props.showGrid
  }

  if (pointCloud && currentModel) {
    const material = pointCloud.material as THREE.PointsMaterial
    material.size = props.pointSize
    const geometry = pointCloud.geometry
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(buildDisplayColors(currentModel, props.colorMode), 3))
    geometry.attributes.color.needsUpdate = true
    pointCloud.visible = props.visualizationMode !== 'cameras'
  }

  if (cameraGroup) {
    cameraGroup.visible = props.visualizationMode !== 'model'
  }
}

function renderLoop() {
  if (!renderer || !scene || !camera || !controls) {
    return
  }
  controls.update()
  renderer.render(scene, camera)
  frameHandle = requestAnimationFrame(renderLoop)
}

function buildCloud(model: ModelPayload) {
  if (!scene) {
    return
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(model.positions, 3))
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(buildDisplayColors(model, props.colorMode), 3))

  const material = new THREE.PointsMaterial({
    size: props.pointSize,
    vertexColors: true,
    sizeAttenuation: true
  })

  pointCloud = new THREE.Points(geometry, material)
  scene.add(pointCloud)
}

function buildCameras(model: ModelPayload) {
  if (!scene || model.cameraPositions.length === 0 || model.cameraRotations.length === 0) {
    return
  }

  const fillPositions: number[] = []
  const linePositions: number[] = []
  const group = new THREE.Group()
  const size = cameraMaterialSize()
  const halfWidth = size * 0.55
  const halfHeight = size * 0.34
  const depth = size

  const localCorners = [
    new THREE.Vector3(-halfWidth, -halfHeight, depth),
    new THREE.Vector3(halfWidth, -halfHeight, depth),
    new THREE.Vector3(halfWidth, halfHeight, depth),
    new THREE.Vector3(-halfWidth, halfHeight, depth)
  ]
  const localTip = new THREE.Vector3(0, 0, 0)

  for (let index = 0; index < model.cameraPositions.length; index += 3) {
    const position = new THREE.Vector3(
      model.cameraPositions[index],
      model.cameraPositions[index + 1],
      model.cameraPositions[index + 2]
    )
    const rotationIndex = (index / 3) * 4
    const quaternion = new THREE.Quaternion(
      model.cameraRotations[rotationIndex],
      model.cameraRotations[rotationIndex + 1],
      model.cameraRotations[rotationIndex + 2],
      model.cameraRotations[rotationIndex + 3]
    )

    const worldCorners = localCorners.map((corner) => corner.clone().applyQuaternion(quaternion).add(position))
    const worldTip = localTip.clone().applyQuaternion(quaternion).add(position)

    fillPositions.push(
      worldCorners[0].x, worldCorners[0].y, worldCorners[0].z,
      worldCorners[1].x, worldCorners[1].y, worldCorners[1].z,
      worldCorners[2].x, worldCorners[2].y, worldCorners[2].z,
      worldCorners[0].x, worldCorners[0].y, worldCorners[0].z,
      worldCorners[2].x, worldCorners[2].y, worldCorners[2].z,
      worldCorners[3].x, worldCorners[3].y, worldCorners[3].z
    )

    const edges = [
      [worldTip, worldCorners[0]],
      [worldTip, worldCorners[1]],
      [worldTip, worldCorners[2]],
      [worldTip, worldCorners[3]],
      [worldCorners[0], worldCorners[1]],
      [worldCorners[1], worldCorners[2]],
      [worldCorners[2], worldCorners[3]],
      [worldCorners[3], worldCorners[0]]
    ]

    for (const [from, to] of edges) {
      linePositions.push(from.x, from.y, from.z, to.x, to.y, to.z)
    }
  }

  const fillGeometry = new THREE.BufferGeometry()
  fillGeometry.setAttribute('position', new THREE.Float32BufferAttribute(fillPositions, 3))
  const fillMaterial = new THREE.MeshBasicMaterial({
    color: '#ff2a12',
    transparent: true,
    opacity: 0.78,
    side: THREE.DoubleSide,
    depthWrite: false
  })

  const lineGeometry = new THREE.BufferGeometry()
  lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3))
  const lineMaterial = new THREE.LineBasicMaterial({
    color: '#e01900',
    transparent: true,
    opacity: 0.95
  })

  group.add(new THREE.Mesh(fillGeometry, fillMaterial))
  group.add(new THREE.LineSegments(lineGeometry, lineMaterial))
  cameraGroup = group
  scene.add(cameraGroup)
}

function initViewer() {
  if (!container.value) {
    return
  }
  disposeViewer()

  const width = container.value.clientWidth
  const height = container.value.clientHeight

  scene = new THREE.Scene()
  scene.background = new THREE.Color(backgroundColor(props.backgroundMode))

  if (props.projectionMode === 'orthographic') {
    camera = new THREE.OrthographicCamera(-6, 6, 6, -6, -200, 400)
  } else {
    camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 500)
  }
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(width, height)

  const hemi = new THREE.HemisphereLight('#fff7ef', '#7c6a56', 1.45)
  const dir = new THREE.DirectionalLight('#ffffff', 1.05)
  dir.position.set(4, 6, 5)
  scene.add(hemi, dir)

  axesHelper = new THREE.AxesHelper(1.6)
  axesHelper.visible = props.showAxes
  scene.add(axesHelper)

  gridHelper = new THREE.GridHelper(10, 20, '#8b7d6d', '#cdbfaa')
  gridHelper.visible = props.showGrid
  scene.add(gridHelper)

  container.value.innerHTML = ''
  container.value.appendChild(renderer.domElement)
  renderer.domElement.addEventListener('dblclick', handleDoubleClick)

  if (props.model) {
    currentModel = props.model
    buildCloud(props.model)
    buildCameras(props.model)
  }

  configureControls()

  if (props.model) {
    fitToModel()
  }

  applyVisualSettings()
  renderLoop()
}

function resizeViewer() {
  if (!container.value || !renderer || !camera) {
    return
  }
  const width = container.value.clientWidth
  const height = container.value.clientHeight
  renderer.setSize(width, height)
  if (camera instanceof THREE.PerspectiveCamera) {
    camera.aspect = width / height
    camera.updateProjectionMatrix()
  } else if (currentModel) {
    updateCameraFrustum(controls?.target ?? modelMetrics(currentModel).center, modelMetrics(currentModel).radius)
  }
}

watch(
  () => props.model,
  () => {
    currentModel = props.model
    initViewer()
  },
  { immediate: true }
)

watch(
  () => props.navigationMode,
  () => {
    initViewer()
  }
)

watch(
  () => props.projectionMode,
  () => {
    initViewer()
  }
)

watch(
  () => props.cameraSize,
  () => {
    initViewer()
  }
)

watch(
  () => props.visualizationMode,
  () => {
    if (!currentModel) {
      return
    }
    fitToModel()
    applyVisualSettings()
  }
)

watch(
  () => [props.pointSize, props.showAxes, props.showGrid, props.colorMode, props.backgroundMode],
  () => {
    applyVisualSettings()
  }
)

defineExpose({
  resetView: () => applyCameraPreset('iso'),
  setFrontView: () => applyCameraPreset('front'),
  setSideView: () => applyCameraPreset('side'),
  setTopView: () => applyCameraPreset('top'),
  fitView: fitToModel,
  recenterView: () => currentModel && recenterOnTarget(modelMetrics(currentModel).center)
})

onMounted(() => {
  initViewer()
  window.addEventListener('resize', resizeViewer)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeViewer)
  renderer?.domElement.removeEventListener('dblclick', handleDoubleClick)
  disposeViewer()
})
</script>

<template>
  <div class="viewer-wrap">
    <div class="viewer-help">
      <span>Doble click: enfocar punto</span>
      <span>Rueda: zoom al cursor</span>
      <span>{{ props.navigationMode === 'map' ? 'Izq: pan, Der: rotar' : 'Izq: rotar, Der: pan' }}</span>
    </div>
    <div ref="container" class="viewer-shell" />
  </div>
</template>

<style scoped>
.viewer-wrap {
  position: relative;
}

.viewer-help {
  position: absolute;
  top: 14px;
  left: 14px;
  z-index: 2;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  max-width: calc(100% - 28px);
}

.viewer-help span {
  padding: 7px 10px;
  border-radius: 999px;
  background: rgba(255, 252, 246, 0.88);
  border: 1px solid rgba(120, 104, 90, 0.16);
  color: #625448;
  font-size: 0.78rem;
  backdrop-filter: blur(10px);
}

.viewer-shell {
  width: 100%;
  height: 68vh;
  min-height: 560px;
  border: 1px solid rgba(120, 104, 90, 0.18);
  border-radius: 24px;
  overflow: hidden;
}

@media (max-width: 900px) {
  .viewer-shell {
    min-height: 420px;
    height: 54vh;
  }
}
</style>
