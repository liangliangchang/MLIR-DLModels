#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0 + d1 + d2 - d4 + 1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d4 * 2)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d1 + d2 * 2)>
#set = affine_set<(d0, d1, d2, d3, d4) : (d0 * 2 + d1 + d3 - 1 >= 0, d0 * -2 - d1 - d3 + 1 >= 0, -d1 - d2 * 2 + 4 >= 0)>
module @funky_sum {
  func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<5xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [3] : tensor<3xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg1 : tensor<3xf32>) outs(%0 : tensor<3xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %2 = tensor.pad %1 low[0] high[1] {
    ^bb0(%arg2: index):
      tensor.yield %cst_0 : f32
    } : tensor<3xf32> to tensor<4xf32>
    %3 = linalg.init_tensor [3] : tensor<3xf32>
    %4 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg0 : tensor<3xf32>) outs(%3 : tensor<3xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<3xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %5 = tensor.pad %4 low[1] high[1] {
    ^bb0(%arg2: index):
      tensor.yield %cst_1 : f32
    } : tensor<3xf32> to tensor<5xf32>
    %6 = linalg.init_tensor [5] : tensor<5xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<5xf32>) -> tensor<5xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["reduction", "window", "window", "reduction", "reduction"]} ins(%5, %2 : tensor<5xf32>, tensor<4xf32>) outs(%7 : tensor<5xf32>) attrs =  {constraints = #set, iterator_ranges = [1, 2, 3, 2, 2]} {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %9 = arith.mulf %arg2, %arg3 : f32
      %10 = arith.addf %arg4, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<5xf32>
    return %8 : tensor<5xf32>
  }
}
