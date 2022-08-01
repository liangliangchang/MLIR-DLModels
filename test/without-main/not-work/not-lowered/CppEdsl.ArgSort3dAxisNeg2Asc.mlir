module @argsort {
  func @main(%arg0: tensor<2x2x2xf32>) -> tensor<2x2x2xi32> {
    %0 = tile.argsort asc %arg0[-2] : (tensor<2x2x2xf32>) -> tensor<2x2x2xi32>
    return %0 : tensor<2x2x2xi32>
  }
}
