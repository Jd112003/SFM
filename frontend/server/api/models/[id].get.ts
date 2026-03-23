import { createError, getQuery, getRouterParam } from 'h3'
import { listAvailableModels, loadModelPointCloud } from '../../utils/model-data'

export default defineEventHandler(async (event) => {
  const id = getRouterParam(event, 'id')
  if (!id) {
    throw createError({ statusCode: 400, statusMessage: 'Missing model id' })
  }

  const maxPointsRaw = Number(getQuery(event).maxPoints ?? 25000)
  const maxPoints = Number.isFinite(maxPointsRaw) ? Math.max(1000, Math.min(60000, Math.floor(maxPointsRaw))) : 25000

  const models = await listAvailableModels()
  const summary = models.find((model) => model.id === id)
  if (!summary) {
    throw createError({ statusCode: 404, statusMessage: `Unknown model: ${id}` })
  }

  const cloud = await loadModelPointCloud(id, maxPoints)
  return {
    ...summary,
    points: cloud.pointCount,
    positions: cloud.positions,
    colors: cloud.colors,
    cameraPositions: cloud.cameraPositions,
    cameraRotations: cloud.cameraRotations,
    bounds: cloud.bounds
  }
})
