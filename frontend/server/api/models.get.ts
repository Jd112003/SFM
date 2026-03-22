import { listAvailableModels } from '../utils/model-data'

export default defineEventHandler(async () => {
  return listAvailableModels()
})
