module @LayerMultipleReturnValues {
  func @main(%arg0: tensor<10x5xf32>) -> (tensor<10x5x5xf32>, tensor<10x5xi32>) {
    %0 = tile.argsort asc %arg0[0] : (tensor<10x5xf32>) -> tensor<10x5xi32>
    %1 = tile.gather %arg0 %0 {OutOfBoundsMode = 0 : i64, axis = 0 : index, batchDims = 0 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 0 : i64, nearestMode = 0 : i64} : (tensor<10x5xf32>, tensor<10x5xi32>) -> tensor<10x5x5xf32>
    return %1, %0 : tensor<10x5x5xf32>, tensor<10x5xi32>
  }
}
