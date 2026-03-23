export interface ModelSummary {
  id: string
  name: string
  objectId: string
  points: number
  originalPoints: number
  filteredModelPath: string
}

export interface ModelPayload {
  id: string
  name: string
  objectId: string
  points: number
  originalPoints: number
  positions: number[]
  colors: number[]
  cameraPositions: number[]
  cameraRotations: number[]
  bounds: {
    min: [number, number, number]
    max: [number, number, number]
  }
}
