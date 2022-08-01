#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 - d1 + 2)>
#map2 = affine_map<(d0, d1) -> (d0 + 1)>
module @cumsum {
  func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [4] : tensor<4xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %2 = tensor.pad %1 low[2] high[0] {
    ^bb0(%arg1: index):
      tensor.yield %cst_0 : f32
    } : tensor<4xf32> to tensor<6xf32>
    %3 = linalg.init_tensor [4] : tensor<4xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<4xf32>) -> tensor<4xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["window", "reduction"]} ins(%2 : tensor<6xf32>) outs(%4 : tensor<4xf32>) attrs =  {iterator_ranges = [3, 3]} {
    ^bb0(%arg1: f32, %arg2: f32):
      %6 = arith.addf %arg2, %arg1 : f32
      linalg.yield %6 : f32
    } -> tensor<4xf32>
    return %5 : tensor<4xf32>
  }
}
