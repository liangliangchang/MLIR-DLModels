module @interpolated_gather_linear {
  func @main(%arg0: tensor<1x6xf32>, %arg1: tensor<12xf32>) -> tensor<1x12xf32> {
    %0 = tile.gather %arg0 %arg1 {OutOfBoundsMode = 0 : i64, axis = 1 : index, batchDims = 0 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 1 : i64, mode = 0 : i64, nearestMode = 0 : i64} : (tensor<1x6xf32>, tensor<12xf32>) -> tensor<1x12xf32>
    return %0 : tensor<1x12xf32>
  }
}
