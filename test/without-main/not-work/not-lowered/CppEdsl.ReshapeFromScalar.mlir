#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module @reshape_from_scalar {
  func @main(%arg0: tensor<i32>) -> tensor<1x1x1xi32> {
    %0 = tensor.expand_shape %arg0 [] : tensor<i32> into tensor<1x1x1xi32>
    %1 = linalg.init_tensor [1, 1, 1] : tensor<1x1x1xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<1x1x1xi32>) outs(%1 : tensor<1x1x1xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<1x1x1xi32>
    return %2 : tensor<1x1x1xi32>
  }
}
