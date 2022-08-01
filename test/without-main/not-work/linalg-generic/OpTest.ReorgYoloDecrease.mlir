#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3 * 2 + d5, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>
module @reorg_yolo {
  func @main(%arg0: tensor<1x4x6x6xi64>) -> tensor<1x1x12x12xi64> {
    %c0_i64 = arith.constant 0 : i64
    %0 = linalg.init_tensor [1, 1, 12, 12] : tensor<1x1x12x12xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x1x12x12xi64>) -> tensor<1x1x12x12xi64>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "window", "window", "window", "window"]} ins(%arg0 : tensor<1x4x6x6xi64>) outs(%1 : tensor<1x1x12x12xi64>) attrs =  {iterator_ranges = [1, 1, 6, 2, 6, 2], name = "reorg_yolo"} {
    ^bb0(%arg1: i64, %arg2: i64):
      linalg.yield %arg1 : i64
    } -> tensor<1x1x12x12xi64>
    return %2 : tensor<1x1x12x12xi64>
  }
}
