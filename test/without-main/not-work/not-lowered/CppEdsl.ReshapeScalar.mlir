#map = affine_map<() -> ()>
module @reshape_scalar {
  func @main(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = linalg.init_tensor [] : tensor<i32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : tensor<i32>) outs(%0 : tensor<i32>) {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<i32>
    return %1 : tensor<i32>
  }
}
