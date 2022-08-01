module @scatter {
  func @main(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x4xf32>) -> tensor<1x4x4x4xf32> {
    %0 = tile.scatter %arg0 %arg1 %arg2 {axis = 0 : index, mode = 3 : i64} : (tensor<1x4x4x4xf32>, tensor<2x3xi32>, tensor<2x4xf32>) -> tensor<1x4x4x4xf32>
    return %0 : tensor<1x4x4x4xf32>
  }
}
