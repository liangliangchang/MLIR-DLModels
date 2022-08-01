module @gatherND {
  func @main(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x1x1xi32>) -> tensor<2x3x1xf32> {
    %0 = tile.gather %arg0 %arg1 {OutOfBoundsMode = 0 : i64, axis = 0 : index, batchDims = 2 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 1 : i64, nearestMode = 0 : i64} : (tensor<2x3x4xf32>, tensor<2x3x1x1xi32>) -> tensor<2x3x1xf32>
    return %0 : tensor<2x3x1xf32>
  }
}
