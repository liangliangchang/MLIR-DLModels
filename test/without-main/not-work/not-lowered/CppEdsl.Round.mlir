#map = affine_map<(d0, d1) -> (d0, d1)>
module @round {
  func @main(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
    %0 = linalg.init_tensor [3, 3] : tensor<3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<3x3xf32>) outs(%0 : tensor<3x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = stdx.round(%arg1) : (f32) -> f32
      linalg.yield %2 : f32
    } -> tensor<3x3xf32>
    return %1 : tensor<3x3xf32>
  }
}
