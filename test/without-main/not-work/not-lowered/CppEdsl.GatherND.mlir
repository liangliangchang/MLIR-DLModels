module @gatherND {
  func @main(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xf32> {
    %0 = tile.gather %arg0 %arg1 {OutOfBoundsMode = 0 : i64, axis = 0 : index, batchDims = 0 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 1 : i64, nearestMode = 0 : i64} : (tensor<2x2x2xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
