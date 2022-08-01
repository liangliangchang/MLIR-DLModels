#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2 + 1)>
module @sort {
  func @main(%arg0: tensor<5x4xf32>) -> tensor<5x4xf32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = linalg.init_tensor [5, 4] : tensor<5x4xi32>
    %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<5x4xi32>) {
    ^bb0(%arg1: i32):
      %18 = linalg.index 0 : index
      %19 = arith.index_cast %18 : index to i32
      linalg.yield %19 : i32
    } -> tensor<5x4xi32>
    %2 = linalg.init_tensor [5, 4, 1] : tensor<5x4x1xi32>
    %3 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<5x4x1xi32>) -> tensor<5x4x1xi32>
    %4 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<5x4xi32>) outs(%3 : tensor<5x4x1xi32>) attrs =  {iterator_ranges = [5, 4, 1], name = "unsqueeze"} {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<5x4x1xi32>
    %5 = linalg.init_tensor [5, 4, 2] : tensor<5x4x2xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<5x4x2xi32>) -> tensor<5x4x2xi32>
    %7 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<5x4x1xi32>) outs(%6 : tensor<5x4x2xi32>) attrs =  {iterator_ranges = [5, 4, 1], name = "concatenate"} {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<5x4x2xi32>
    %8 = tile.argsort asc %arg0[1] : (tensor<5x4xf32>) -> tensor<5x4xi32>
    %9 = linalg.init_tensor [5, 4, 1] : tensor<5x4x1xi32>
    %10 = linalg.fill ins(%c0_i32 : i32) outs(%9 : tensor<5x4x1xi32>) -> tensor<5x4x1xi32>
    %11 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<5x4xi32>) outs(%10 : tensor<5x4x1xi32>) attrs =  {iterator_ranges = [5, 4, 1], name = "unsqueeze"} {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<5x4x1xi32>
    %12 = linalg.init_tensor [5, 4, 2] : tensor<5x4x2xi32>
    %13 = linalg.fill ins(%c0_i32 : i32) outs(%12 : tensor<5x4x2xi32>) -> tensor<5x4x2xi32>
    %14 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "window"]} ins(%11 : tensor<5x4x1xi32>) outs(%13 : tensor<5x4x2xi32>) attrs =  {iterator_ranges = [5, 4, 1], name = "concatenate"} {
    ^bb0(%arg1: i32, %arg2: i32):
      linalg.yield %arg1 : i32
    } -> tensor<5x4x2xi32>
    %15 = linalg.init_tensor [5, 4, 2] : tensor<5x4x2xi32>
    %16 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7, %14 : tensor<5x4x2xi32>, tensor<5x4x2xi32>) outs(%15 : tensor<5x4x2xi32>) {
    ^bb0(%arg1: i32, %arg2: i32, %arg3: i32):
      %18 = arith.addi %arg1, %arg2 : i32
      linalg.yield %18 : i32
    } -> tensor<5x4x2xi32>
    %17 = tile.gather %arg0 %16 {OutOfBoundsMode = 0 : i64, axis = 0 : index, batchDims = 0 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 1 : i64, nearestMode = 0 : i64} : (tensor<5x4xf32>, tensor<5x4x2xi32>) -> tensor<5x4xf32>
    return %17 : tensor<5x4xf32>
  }
}
