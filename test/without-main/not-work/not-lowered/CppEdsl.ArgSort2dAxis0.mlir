module @argsort {
  func @main(%arg0: tensor<5x4xf32>) -> tensor<5x4xi32> {
    %0 = tile.argsort asc %arg0[0] : (tensor<5x4xf32>) -> tensor<5x4xi32>
    return %0 : tensor<5x4xi32>
  }
}
