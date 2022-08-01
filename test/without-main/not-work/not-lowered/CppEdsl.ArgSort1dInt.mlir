module @argsort {
  func @main(%arg0: tensor<20xi32>) -> tensor<20xi32> {
    %0 = tile.argsort asc %arg0[0] : (tensor<20xi32>) -> tensor<20xi32>
    return %0 : tensor<20xi32>
  }
}
