#map0 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0 + 1, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0 + 2, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1 + 2, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, 1, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d1 + 4, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1, d2 + 1)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, 2)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d1, d2 + 4)>
module @explicit_padding {
  func @main(%arg0: tensor<1x2x3xf32>) -> tensor<4x5x7xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [1, 2, 3] : tensor<1x2x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x2x3xf32>) outs(%1 : tensor<1x2x3xf32>) attrs =  {iterator_ranges = [1, 2, 3], name = "explicit_padding"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<1x2x3xf32>
    %3 = linalg.init_tensor [4, 2, 3] : tensor<4x2x3xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x2x3xf32>) outs(%4 : tensor<4x2x3xf32>) attrs =  {iterator_ranges = [1, 2, 3], name = "concatenate"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x2x3xf32>
    %6 = linalg.init_tensor [4, 2, 3] : tensor<4x2x3xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["window", "parallel", "parallel"]} ins(%arg0 : tensor<1x2x3xf32>) outs(%7 : tensor<4x2x3xf32>) attrs =  {iterator_ranges = [1, 2, 3], name = "concatenate"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x2x3xf32>
    %9 = linalg.init_tensor [4, 2, 3] : tensor<4x2x3xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %8 : tensor<4x2x3xf32>, tensor<4x2x3xf32>) outs(%9 : tensor<4x2x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %57 = arith.addf %arg1, %arg2 : f32
      linalg.yield %57 : f32
    } -> tensor<4x2x3xf32>
    %11 = linalg.init_tensor [2, 2, 3] : tensor<2x2x3xf32>
    %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>
    %13 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x2x3xf32>) outs(%12 : tensor<2x2x3xf32>) attrs =  {iterator_ranges = [2, 2, 3], name = "explicit_padding"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<2x2x3xf32>
    %14 = linalg.init_tensor [4, 2, 3] : tensor<4x2x3xf32>
    %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map3], iterator_types = ["window", "parallel", "parallel"]} ins(%13 : tensor<2x2x3xf32>) outs(%15 : tensor<4x2x3xf32>) attrs =  {iterator_ranges = [2, 2, 3], name = "concatenate"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x2x3xf32>
    %17 = linalg.init_tensor [4, 2, 3] : tensor<4x2x3xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %16 : tensor<4x2x3xf32>, tensor<4x2x3xf32>) outs(%17 : tensor<4x2x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %57 = arith.addf %arg1, %arg2 : f32
      linalg.yield %57 : f32
    } -> tensor<4x2x3xf32>
    %19 = linalg.init_tensor [4, 2, 3] : tensor<4x2x3xf32>
    %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
    %21 = linalg.generic {indexing_maps = [#map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18 : tensor<4x2x3xf32>) outs(%20 : tensor<4x2x3xf32>) attrs =  {iterator_ranges = [4, 2, 3], name = "explicit_padding"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x2x3xf32>
    %22 = linalg.init_tensor [4, 5, 3] : tensor<4x5x3xf32>
    %23 = linalg.fill ins(%cst : f32) outs(%22 : tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21 : tensor<4x2x3xf32>) outs(%23 : tensor<4x5x3xf32>) attrs =  {iterator_ranges = [4, 2, 3], name = "concatenate"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x5x3xf32>
    %25 = linalg.init_tensor [4, 5, 3] : tensor<4x5x3xf32>
    %26 = linalg.fill ins(%cst : f32) outs(%25 : tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    %27 = linalg.generic {indexing_maps = [#map1, #map5], iterator_types = ["parallel", "window", "parallel"]} ins(%18 : tensor<4x2x3xf32>) outs(%26 : tensor<4x5x3xf32>) attrs =  {iterator_ranges = [4, 2, 3], name = "concatenate"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x5x3xf32>
    %28 = linalg.init_tensor [4, 5, 3] : tensor<4x5x3xf32>
    %29 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %27 : tensor<4x5x3xf32>, tensor<4x5x3xf32>) outs(%28 : tensor<4x5x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %57 = arith.addf %arg1, %arg2 : f32
      linalg.yield %57 : f32
    } -> tensor<4x5x3xf32>
    %30 = linalg.init_tensor [4, 1, 3] : tensor<4x1x3xf32>
    %31 = linalg.fill ins(%cst : f32) outs(%30 : tensor<4x1x3xf32>) -> tensor<4x1x3xf32>
    %32 = linalg.generic {indexing_maps = [#map6, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18 : tensor<4x2x3xf32>) outs(%31 : tensor<4x1x3xf32>) attrs =  {iterator_ranges = [4, 1, 3], name = "explicit_padding"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x1x3xf32>
    %33 = linalg.init_tensor [4, 5, 3] : tensor<4x5x3xf32>
    %34 = linalg.fill ins(%cst : f32) outs(%33 : tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    %35 = linalg.generic {indexing_maps = [#map1, #map7], iterator_types = ["parallel", "window", "parallel"]} ins(%32 : tensor<4x1x3xf32>) outs(%34 : tensor<4x5x3xf32>) attrs =  {iterator_ranges = [4, 1, 3], name = "concatenate"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x5x3xf32>
    %36 = linalg.init_tensor [4, 5, 3] : tensor<4x5x3xf32>
    %37 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %35 : tensor<4x5x3xf32>, tensor<4x5x3xf32>) outs(%36 : tensor<4x5x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %57 = arith.addf %arg1, %arg2 : f32
      linalg.yield %57 : f32
    } -> tensor<4x5x3xf32>
    %38 = linalg.init_tensor [4, 5, 1] : tensor<4x5x1xf32>
    %39 = linalg.fill ins(%cst : f32) outs(%38 : tensor<4x5x1xf32>) -> tensor<4x5x1xf32>
    %40 = linalg.generic {indexing_maps = [#map8, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37 : tensor<4x5x3xf32>) outs(%39 : tensor<4x5x1xf32>) attrs =  {iterator_ranges = [4, 5, 1], name = "explicit_padding"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x5x1xf32>
    %41 = linalg.init_tensor [4, 5, 7] : tensor<4x5x7xf32>
    %42 = linalg.fill ins(%cst : f32) outs(%41 : tensor<4x5x7xf32>) -> tensor<4x5x7xf32>
    %43 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40 : tensor<4x5x1xf32>) outs(%42 : tensor<4x5x7xf32>) attrs =  {iterator_ranges = [4, 5, 1], name = "concatenate"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x5x7xf32>
    %44 = linalg.init_tensor [4, 5, 7] : tensor<4x5x7xf32>
    %45 = linalg.fill ins(%cst : f32) outs(%44 : tensor<4x5x7xf32>) -> tensor<4x5x7xf32>
    %46 = linalg.generic {indexing_maps = [#map1, #map9], iterator_types = ["parallel", "parallel", "window"]} ins(%37 : tensor<4x5x3xf32>) outs(%45 : tensor<4x5x7xf32>) attrs =  {iterator_ranges = [4, 5, 3], name = "concatenate"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x5x7xf32>
    %47 = linalg.init_tensor [4, 5, 7] : tensor<4x5x7xf32>
    %48 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %46 : tensor<4x5x7xf32>, tensor<4x5x7xf32>) outs(%47 : tensor<4x5x7xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %57 = arith.addf %arg1, %arg2 : f32
      linalg.yield %57 : f32
    } -> tensor<4x5x7xf32>
    %49 = linalg.init_tensor [4, 5, 3] : tensor<4x5x3xf32>
    %50 = linalg.fill ins(%cst : f32) outs(%49 : tensor<4x5x3xf32>) -> tensor<4x5x3xf32>
    %51 = linalg.generic {indexing_maps = [#map10, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37 : tensor<4x5x3xf32>) outs(%50 : tensor<4x5x3xf32>) attrs =  {iterator_ranges = [4, 5, 3], name = "explicit_padding"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x5x3xf32>
    %52 = linalg.init_tensor [4, 5, 7] : tensor<4x5x7xf32>
    %53 = linalg.fill ins(%cst : f32) outs(%52 : tensor<4x5x7xf32>) -> tensor<4x5x7xf32>
    %54 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "window"]} ins(%51 : tensor<4x5x3xf32>) outs(%53 : tensor<4x5x7xf32>) attrs =  {iterator_ranges = [4, 5, 3], name = "concatenate"} {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<4x5x7xf32>
    %55 = linalg.init_tensor [4, 5, 7] : tensor<4x5x7xf32>
    %56 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %54 : tensor<4x5x7xf32>, tensor<4x5x7xf32>) outs(%55 : tensor<4x5x7xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %57 = arith.addf %arg1, %arg2 : f32
      linalg.yield %57 : f32
    } -> tensor<4x5x7xf32>
    return %56 : tensor<4x5x7xf32>
  }
}
