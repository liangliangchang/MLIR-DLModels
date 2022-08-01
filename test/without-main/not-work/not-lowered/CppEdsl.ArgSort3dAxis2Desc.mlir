module @argsort {
  func @main(%arg0: tensor<3x4x5xf32>) -> tensor<3x4x5xi32> {
    %0 = tile.argsort desc %arg0[2] : (tensor<3x4x5xf32>) -> tensor<3x4x5xi32>
    return %0 : tensor<3x4x5xi32>
  }
}
