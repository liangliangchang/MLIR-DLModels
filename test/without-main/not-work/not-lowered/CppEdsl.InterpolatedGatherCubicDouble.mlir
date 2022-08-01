module @interpolated_gather_cubic_double {
  func @main(%arg0: tensor<1x6xf64>, %arg1: tensor<12xf64>) -> tensor<1x12xf64> {
    %0 = tile.gather %arg0 %arg1 {OutOfBoundsMode = 0 : i64, axis = 1 : index, batchDims = 0 : index, cubeCoeff = -5.000000e-01 : f64, interpolationMode = 2 : i64, mode = 0 : i64, nearestMode = 0 : i64} : (tensor<1x6xf64>, tensor<12xf64>) -> tensor<1x12xf64>
    return %0 : tensor<1x12xf64>
  }
}
