module @interpolated_gather_nearest {
  func @main(%arg0: tensor<1x6xf32>, %arg1: tensor<9xf32>) -> tensor<1x9xf32> {
    %0 = tile.gather %arg0 %arg1 {OutOfBoundsMode = 0 : i64, axis = 1 : index, batchDims = 0 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 0 : i64, mode = 0 : i64, nearestMode = 1 : i64} : (tensor<1x6xf32>, tensor<9xf32>) -> tensor<1x9xf32>
    return %0 : tensor<1x9xf32>
  }
}
