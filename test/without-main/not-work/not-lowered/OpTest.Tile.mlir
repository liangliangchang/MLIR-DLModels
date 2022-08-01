#map0 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 10 + d1, d2 * 20 + d3)>
module @tile {
  func @main(%arg0: tensor<10x20xf32>) -> tensor<50x80xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [50, 80] : tensor<50x80xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<50x80xf32>) -> tensor<50x80xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["window", "window", "window", "window"]} ins(%arg0 : tensor<10x20xf32>) outs(%1 : tensor<50x80xf32>) attrs =  {iterator_ranges = [5, 10, 4, 20], name = "tile"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<50x80xf32>
    return %2 : tensor<50x80xf32>
  }
}
