#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0 + 2, d1)>
#map3 = affine_map<(d0, d1) -> ()>
#map4 = affine_map<(d0) -> ()>
#map5 = affine_map<(d0) -> (0)>
#map6 = affine_map<(d0, d1) -> (d0)>
#map7 = affine_map<(d0, d1) -> (0)>
module @gatherND {
  func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<3x1xf32>) -> tensor<3x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = linalg.init_tensor [1] : tensor<1xi32>
    %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%0 : tensor<1xi32>) {
    ^bb0(%arg2: i32):
      %52 = linalg.index 0 : index
      %53 = arith.index_cast %52 : index to i32
      linalg.yield %53 : i32
    } -> tensor<1xi32>
    %2 = linalg.init_tensor [1] : tensor<1xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%1 : tensor<1xi32>) outs(%2 : tensor<1xf32>) {
    ^bb0(%arg2: i32, %arg3: f32):
      %52 = arith.sitofp %arg2 : i32 to f32
      linalg.yield %52 : f32
    } -> tensor<1xf32>
    %4 = linalg.init_tensor [6, 2] : tensor<6x2xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<6x2xf32>) -> tensor<6x2xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["window", "parallel"]} ins(%arg0 : tensor<2x2xf32>) outs(%5 : tensor<6x2xf32>) attrs =  {iterator_ranges = [2, 2], name = "explicit_padding"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<6x2xf32>
    %7 = linalg.init_tensor [3, 1] : tensor<3x1xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %cst_0 : tensor<3x1xf32>, f32) outs(%7 : tensor<3x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.addf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<3x1xf32>
    %9 = tile.gather %8 %1 {OutOfBoundsMode = 0 : i64, axis = 1 : index, batchDims = 0 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 0 : i64, nearestMode = 0 : i64} : (tensor<3x1xf32>, tensor<1xi32>) -> tensor<3x1xf32>
    %10 = linalg.init_tensor [3, 1] : tensor<3x1xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<3x1xf32>) outs(%10 : tensor<3x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %52 = math.floor %arg2 : f32
      linalg.yield %52 : f32
    } -> tensor<3x1xf32>
    %12 = linalg.init_tensor [3, 1] : tensor<3x1xf32>
    %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<3x1xf32>) -> tensor<3x1xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<3x1xf32>) outs(%13 : tensor<3x1xf32>) attrs =  {iterator_ranges = [3, 1], name = "concatenate"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<3x1xf32>
    %15 = linalg.init_tensor [3, 1] : tensor<3x1xi32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<3x1xf32>) outs(%15 : tensor<3x1xi32>) {
    ^bb0(%arg2: f32, %arg3: i32):
      %52 = arith.fptosi %arg2 : f32 to i32
      linalg.yield %52 : i32
    } -> tensor<3x1xi32>
    %17 = tile.gather %6 %16 {OutOfBoundsMode = 0 : i64, axis = 0 : index, batchDims = 0 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 1 : i64, nearestMode = 0 : i64} : (tensor<6x2xf32>, tensor<3x1xi32>) -> tensor<3x2xf32>
    %18 = linalg.init_tensor [1] : tensor<1xf32>
    %19 = linalg.generic {indexing_maps = [#map0, #map4, #map0], iterator_types = ["parallel"]} ins(%3, %cst_1 : tensor<1xf32>, f32) outs(%18 : tensor<1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.addf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<1xf32>
    %20 = linalg.init_tensor [3, 1] : tensor<3x1xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%11, %cst_1 : tensor<3x1xf32>, f32) outs(%20 : tensor<3x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.addf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<3x1xf32>
    %22 = linalg.init_tensor [3, 1] : tensor<3x1xf32>
    %23 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%21, %9 : tensor<3x1xf32>, tensor<3x1xf32>) outs(%22 : tensor<3x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.subf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<3x1xf32>
    %24 = tensor.collapse_shape %23 [[0, 1]] : tensor<3x1xf32> into tensor<3xf32>
    %25 = linalg.init_tensor [3] : tensor<3xf32>
    %26 = linalg.generic {indexing_maps = [#map5, #map0, #map0], iterator_types = ["parallel"]} ins(%19, %24 : tensor<1xf32>, tensor<3xf32>) outs(%25 : tensor<3xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<3xf32>
    %27 = linalg.init_tensor [3, 2] : tensor<3x2xf32>
    %28 = linalg.fill ins(%cst : f32) outs(%27 : tensor<3x2xf32>) -> tensor<3x2xf32>
    %29 = linalg.generic {indexing_maps = [#map6, #map1], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<3xf32>) outs(%28 : tensor<3x2xf32>) attrs =  {iterator_ranges = [3, 2], name = "broadcast"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<3x2xf32>
    %30 = linalg.init_tensor [3, 2] : tensor<3x2xf32>
    %31 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%17, %29 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%30 : tensor<3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<3x2xf32>
    %32 = linalg.init_tensor [3, 2] : tensor<3x2xf32>
    %33 = linalg.generic {indexing_maps = [#map7, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %31 : tensor<1xf32>, tensor<3x2xf32>) outs(%32 : tensor<3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.addf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<3x2xf32>
    %34 = linalg.init_tensor [3, 1] : tensor<3x1xf32>
    %35 = linalg.fill ins(%cst : f32) outs(%34 : tensor<3x1xf32>) -> tensor<3x1xf32>
    %36 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%21 : tensor<3x1xf32>) outs(%35 : tensor<3x1xf32>) attrs =  {iterator_ranges = [3, 1], name = "concatenate"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<3x1xf32>
    %37 = linalg.init_tensor [3, 1] : tensor<3x1xi32>
    %38 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%36 : tensor<3x1xf32>) outs(%37 : tensor<3x1xi32>) {
    ^bb0(%arg2: f32, %arg3: i32):
      %52 = arith.fptosi %arg2 : f32 to i32
      linalg.yield %52 : i32
    } -> tensor<3x1xi32>
    %39 = tile.gather %6 %38 {OutOfBoundsMode = 0 : i64, axis = 0 : index, batchDims = 0 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 1 : i64, nearestMode = 0 : i64} : (tensor<6x2xf32>, tensor<3x1xi32>) -> tensor<3x2xf32>
    %40 = linalg.init_tensor [3, 1] : tensor<3x1xf32>
    %41 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %11 : tensor<3x1xf32>, tensor<3x1xf32>) outs(%40 : tensor<3x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.subf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<3x1xf32>
    %42 = tensor.collapse_shape %41 [[0, 1]] : tensor<3x1xf32> into tensor<3xf32>
    %43 = linalg.init_tensor [3] : tensor<3xf32>
    %44 = linalg.generic {indexing_maps = [#map5, #map0, #map0], iterator_types = ["parallel"]} ins(%19, %42 : tensor<1xf32>, tensor<3xf32>) outs(%43 : tensor<3xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<3xf32>
    %45 = linalg.init_tensor [3, 2] : tensor<3x2xf32>
    %46 = linalg.fill ins(%cst : f32) outs(%45 : tensor<3x2xf32>) -> tensor<3x2xf32>
    %47 = linalg.generic {indexing_maps = [#map6, #map1], iterator_types = ["parallel", "parallel"]} ins(%44 : tensor<3xf32>) outs(%46 : tensor<3x2xf32>) attrs =  {iterator_ranges = [3, 2], name = "broadcast"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<3x2xf32>
    %48 = linalg.init_tensor [3, 2] : tensor<3x2xf32>
    %49 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%39, %47 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%48 : tensor<3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<3x2xf32>
    %50 = linalg.init_tensor [3, 2] : tensor<3x2xf32>
    %51 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%33, %49 : tensor<3x2xf32>, tensor<3x2xf32>) outs(%50 : tensor<3x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %52 = arith.addf %arg2, %arg3 : f32
      linalg.yield %52 : f32
    } -> tensor<3x2xf32>
    return %51 : tensor<3x2xf32>
  }
}
