module @scatter {
  func @main(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi32>, %arg2: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> {
    %0 = tile.scatter %arg0 %arg1 %arg2 {axis = 0 : index, mode = 3 : i64} : (tensor<4x4x4xf32>, tensor<2x1xi32>, tensor<2x4x4xf32>) -> tensor<4x4x4xf32>
    return %0 : tensor<4x4x4xf32>
  }
}
