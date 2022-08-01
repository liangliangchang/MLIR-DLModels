module @prng {
  func @main(%arg0: tensor<1x3xi32> {stdx.const}) -> (tensor<2x3xf32>, tensor<1x3xi32>) {
    %result, %new_state = tile.prng %arg0 : (tensor<1x3xi32>) -> (tensor<2x3xf32>, tensor<1x3xi32>)
    return %result, %new_state : tensor<2x3xf32>, tensor<1x3xi32>
  }
}
