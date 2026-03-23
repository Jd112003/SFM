import { promises as fs } from 'node:fs'
import { resolve } from 'node:path'

interface ParsedPointCloud {
  positions: number[]
  colors: number[]
  cameraPositions: number[]
  cameraRotations: number[]
  pointCount: number
  bounds: {
    min: [number, number, number]
    max: [number, number, number]
  }
}

interface ModelEntry {
  id: string
  name: string
  objectId: string
  filteredModelPath: string
  filteredPointsPath: string
  originalPointsPath: string
}

const modelCache = new Map<string, ParsedPointCloud>()

function quaternionToRotationMatrix(qw: number, qx: number, qy: number, qz: number): number[][] {
  return [
    [
      1 - 2 * (qy * qy + qz * qz),
      2 * (qx * qy - qz * qw),
      2 * (qx * qz + qy * qw)
    ],
    [
      2 * (qx * qy + qz * qw),
      1 - 2 * (qx * qx + qz * qz),
      2 * (qy * qz - qx * qw)
    ],
    [
      2 * (qx * qz - qy * qw),
      2 * (qy * qz + qx * qw),
      1 - 2 * (qx * qx + qy * qy)
    ]
  ]
}

function cameraCenterFromPose(qw: number, qx: number, qy: number, qz: number, tx: number, ty: number, tz: number) {
  const rotation = quaternionToRotationMatrix(qw, qx, qy, qz)
  const translation = [tx, ty, tz]

  return [
    -(rotation[0][0] * translation[0] + rotation[1][0] * translation[1] + rotation[2][0] * translation[2]),
    -(rotation[0][1] * translation[0] + rotation[1][1] * translation[1] + rotation[2][1] * translation[2]),
    -(rotation[0][2] * translation[0] + rotation[1][2] * translation[1] + rotation[2][2] * translation[2])
  ] as const
}

async function loadCameraPoses(imagesPath: string): Promise<{ cameraPositions: number[], cameraRotations: number[] }> {
  const content = await fs.readFile(imagesPath, 'utf8')
  const rows = content
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))

  const cameraPositions: number[] = []
  const cameraRotations: number[] = []

  for (let index = 0; index < rows.length; index += 2) {
    const parts = rows[index]?.split(/\s+/) ?? []
    if (parts.length < 8) {
      continue
    }

    const qw = Number(parts[1])
    const qx = Number(parts[2])
    const qy = Number(parts[3])
    const qz = Number(parts[4])
    const tx = Number(parts[5])
    const ty = Number(parts[6])
    const tz = Number(parts[7])

    if ([qw, qx, qy, qz, tx, ty, tz].some((value) => !Number.isFinite(value))) {
      continue
    }

    const [x, y, z] = cameraCenterFromPose(qw, qx, qy, qz, tx, ty, tz)
    cameraPositions.push(x, y, z)
    cameraRotations.push(-qx, -qy, -qz, qw)
  }

  return { cameraPositions, cameraRotations }
}

function outputsRoot(): string {
  return process.env.SFM_OUTPUTS_ROOT || resolve(process.cwd(), '..', 'outputs')
}

async function countPoints(pointsPath: string): Promise<number> {
  const content = await fs.readFile(pointsPath, 'utf8')
  let count = 0
  for (const line of content.split('\n')) {
    const trimmed = line.trim()
    if (trimmed && !trimmed.startsWith('#')) {
      count += 1
    }
  }
  return count
}

export async function listAvailableModels() {
  const entries = await fs.readdir(outputsRoot(), { withFileTypes: true })
  const models: ModelEntry[] = []

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue
    }
    const objectId = entry.name
    const filteredModelPath = resolve(outputsRoot(), objectId, 'reconstruction', 'filtered_text_model')
    const filteredPointsPath = resolve(filteredModelPath, 'points3D.txt')
    const originalPointsPath = resolve(outputsRoot(), objectId, 'reconstruction', 'text_model', 'points3D.txt')

    try {
      await fs.access(filteredPointsPath)
      await fs.access(originalPointsPath)
    } catch {
      continue
    }

    models.push({
      id: objectId,
      name: objectId.charAt(0).toUpperCase() + objectId.slice(1),
      objectId,
      filteredModelPath,
      filteredPointsPath,
      originalPointsPath
    })
  }

  const summaries = await Promise.all(models.map(async (model) => ({
    id: model.id,
    name: model.name,
    objectId: model.objectId,
    filteredModelPath: model.filteredModelPath,
    points: await countPoints(model.filteredPointsPath),
    originalPoints: await countPoints(model.originalPointsPath)
  })))

  return summaries.sort((a, b) => a.name.localeCompare(b.name))
}

function buildDeterministicSample<T>(items: T[], maxPoints: number): T[] {
  if (items.length <= maxPoints) {
    return items
  }
  const stride = items.length / maxPoints
  const sample: T[] = []
  for (let i = 0; i < maxPoints; i += 1) {
    sample.push(items[Math.floor(i * stride)])
  }
  return sample
}

export async function loadModelPointCloud(objectId: string, maxPoints: number) {
  const cacheKey = `${objectId}:${maxPoints}`
  const cached = modelCache.get(cacheKey)
  if (cached) {
    return cached
  }

  const pointsPath = resolve(outputsRoot(), objectId, 'reconstruction', 'filtered_text_model', 'points3D.txt')
  const imagesPath = resolve(outputsRoot(), objectId, 'reconstruction', 'filtered_text_model', 'images.txt')
  const content = await fs.readFile(pointsPath, 'utf8')
  const { cameraPositions, cameraRotations } = await loadCameraPoses(imagesPath)
  const rows = content
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))

  const sampledRows = buildDeterministicSample(rows, maxPoints)
  const positions: number[] = []
  const colors: number[] = []

  const boundsMin: [number, number, number] = [Infinity, Infinity, Infinity]
  const boundsMax: [number, number, number] = [-Infinity, -Infinity, -Infinity]

  for (const row of sampledRows) {
    const parts = row.split(/\s+/)
    if (parts.length < 7) {
      continue
    }
    const x = Number(parts[1])
    const y = Number(parts[2])
    const z = Number(parts[3])
    const r = Number(parts[4]) / 255
    const g = Number(parts[5]) / 255
    const b = Number(parts[6]) / 255

    positions.push(x, y, z)
    colors.push(r, g, b)

    boundsMin[0] = Math.min(boundsMin[0], x)
    boundsMin[1] = Math.min(boundsMin[1], y)
    boundsMin[2] = Math.min(boundsMin[2], z)
    boundsMax[0] = Math.max(boundsMax[0], x)
    boundsMax[1] = Math.max(boundsMax[1], y)
    boundsMax[2] = Math.max(boundsMax[2], z)
  }

  const parsed: ParsedPointCloud = {
    positions,
    colors,
    cameraPositions,
    cameraRotations,
    pointCount: positions.length / 3,
    bounds: {
      min: boundsMin,
      max: boundsMax
    }
  }
  modelCache.set(cacheKey, parsed)
  return parsed
}
