module @interpolated_gather_cubic_half {
  func @main(%arg0: tensor<1x6xf16>, %arg1: tensor<12xf16>) -> tensor<1x12xf16> {
    %0 = tile.gather %arg0 %arg1 {OutOfBoundsMode = 0 : i64, axis = 1 : index, batchDims = 0 : index, cubeCoeff = -5.000000e-01 : f64, interpolationMode = 2 : i64, mode = 0 : i64, nearestMode = 0 : i64} : (tensor<1x6xf16>, tensor<12xf16>) -> tensor<1x12xf16>
    return %0 : tensor<1x12xf16>
  }
}
