#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1 + 2, d2 + 2)>
#map3 = affine_map<(d0, d1, d2) -> ()>
#map4 = affine_map<(d0) -> ()>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2 + 1)>
#map6 = affine_map<(d0, d1) -> (d1)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<(d0, d1) -> (d0, 0)>
module @gatherND {
  func @main(%arg0: tensor<4x3x2xf32>, %arg1: tensor<4x1x2xf32>) -> tensor<4x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = linalg.init_tensor [1] : tensor<1xi32>
    %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel"]} outs(%0 : tensor<1xi32>) {
    ^bb0(%arg2: i32):
      %107 = linalg.index 0 : index
      %108 = arith.index_cast %107 : index to i32
      linalg.yield %108 : i32
    } -> tensor<1xi32>
    %2 = linalg.init_tensor [1] : tensor<1xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%1 : tensor<1xi32>) outs(%2 : tensor<1xf32>) {
    ^bb0(%arg2: i32, %arg3: f32):
      %107 = arith.sitofp %arg2 : i32 to f32
      linalg.yield %107 : f32
    } -> tensor<1xf32>
    %4 = linalg.init_tensor [4, 7, 6] : tensor<4x7x6xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<4x7x6xf32>) -> tensor<4x7x6xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "window", "window"]} ins(%arg0 : tensor<4x3x2xf32>) outs(%5 : tensor<4x7x6xf32>) attrs =  {iterator_ranges = [4, 3, 2], name = "explicit_padding"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<4x7x6xf32>
    %7 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1, %cst_0 : tensor<4x1x2xf32>, f32) outs(%7 : tensor<4x1x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x2xf32>
    %9 = tile.gather %8 %1 {OutOfBoundsMode = 0 : i64, axis = 2 : index, batchDims = 0 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 0 : i64, nearestMode = 0 : i64} : (tensor<4x1x2xf32>, tensor<1xi32>) -> tensor<4x1x1xf32>
    %10 = linalg.init_tensor [4, 1, 1] : tensor<4x1x1xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<4x1x1xf32>) outs(%10 : tensor<4x1x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %107 = math.floor %arg2 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x1xf32>
    %12 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xf32>
    %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<4x1x2xf32>) -> tensor<4x1x2xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11 : tensor<4x1x1xf32>) outs(%13 : tensor<4x1x2xf32>) attrs =  {iterator_ranges = [4, 1, 1], name = "concatenate"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<4x1x2xf32>
    %15 = linalg.init_tensor [1] : tensor<1xi32>
    %16 = linalg.generic {indexing_maps = [#map0, #map4, #map0], iterator_types = ["parallel"]} ins(%1, %c1_i32 : tensor<1xi32>, i32) outs(%15 : tensor<1xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):
      %107 = arith.addi %arg2, %arg3 : i32
      linalg.yield %107 : i32
    } -> tensor<1xi32>
    %17 = tile.gather %8 %16 {OutOfBoundsMode = 0 : i64, axis = 2 : index, batchDims = 0 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 0 : i64, nearestMode = 0 : i64} : (tensor<4x1x2xf32>, tensor<1xi32>) -> tensor<4x1x1xf32>
    %18 = linalg.init_tensor [4, 1, 1] : tensor<4x1x1xf32>
    %19 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17 : tensor<4x1x1xf32>) outs(%18 : tensor<4x1x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %107 = math.floor %arg2 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x1xf32>
    %20 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xf32>
    %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<4x1x2xf32>) -> tensor<4x1x2xf32>
    %22 = linalg.generic {indexing_maps = [#map1, #map5], iterator_types = ["parallel", "parallel", "window"]} ins(%19 : tensor<4x1x1xf32>) outs(%21 : tensor<4x1x2xf32>) attrs =  {iterator_ranges = [4, 1, 1], name = "concatenate"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<4x1x2xf32>
    %23 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %22 : tensor<4x1x2xf32>, tensor<4x1x2xf32>) outs(%23 : tensor<4x1x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x2xf32>
    %25 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xi32>
    %26 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24 : tensor<4x1x2xf32>) outs(%25 : tensor<4x1x2xi32>) {
    ^bb0(%arg2: f32, %arg3: i32):
      %107 = arith.fptosi %arg2 : f32 to i32
      linalg.yield %107 : i32
    } -> tensor<4x1x2xi32>
    %27 = tile.gather %6 %26 {OutOfBoundsMode = 0 : i64, axis = 0 : index, batchDims = 1 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 1 : i64, nearestMode = 0 : i64} : (tensor<4x7x6xf32>, tensor<4x1x2xi32>) -> tensor<4x1xf32>
    %28 = linalg.init_tensor [1] : tensor<1xf32>
    %29 = linalg.generic {indexing_maps = [#map0, #map4, #map0], iterator_types = ["parallel"]} ins(%3, %cst_1 : tensor<1xf32>, f32) outs(%28 : tensor<1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<1xf32>
    %30 = linalg.init_tensor [4, 1, 1] : tensor<4x1x1xf32>
    %31 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %cst_1 : tensor<4x1x1xf32>, f32) outs(%30 : tensor<4x1x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x1xf32>
    %32 = linalg.init_tensor [4, 1, 1] : tensor<4x1x1xf32>
    %33 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %9 : tensor<4x1x1xf32>, tensor<4x1x1xf32>) outs(%32 : tensor<4x1x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.subf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x1xf32>
    %34 = tensor.collapse_shape %33 [[0], [1, 2]] : tensor<4x1x1xf32> into tensor<4x1xf32>
    %35 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %36 = linalg.generic {indexing_maps = [#map6, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%29, %34 : tensor<1xf32>, tensor<4x1xf32>) outs(%35 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %37 = linalg.init_tensor [4, 1, 1] : tensor<4x1x1xf32>
    %38 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %cst_1 : tensor<4x1x1xf32>, f32) outs(%37 : tensor<4x1x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x1xf32>
    %39 = linalg.init_tensor [4, 1, 1] : tensor<4x1x1xf32>
    %40 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %17 : tensor<4x1x1xf32>, tensor<4x1x1xf32>) outs(%39 : tensor<4x1x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.subf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x1xf32>
    %41 = tensor.collapse_shape %40 [[0], [1, 2]] : tensor<4x1x1xf32> into tensor<4x1xf32>
    %42 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %43 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%36, %41 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%42 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %44 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %45 = linalg.fill ins(%cst : f32) outs(%44 : tensor<4x1xf32>) -> tensor<4x1xf32>
    %46 = linalg.generic {indexing_maps = [#map8, #map7], iterator_types = ["parallel", "parallel"]} ins(%43 : tensor<4x1xf32>) outs(%45 : tensor<4x1xf32>) attrs =  {iterator_ranges = [4, 1], name = "broadcast"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<4x1xf32>
    %47 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %48 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%27, %46 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%47 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %49 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %50 = linalg.generic {indexing_maps = [#map6, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%3, %48 : tensor<1xf32>, tensor<4x1xf32>) outs(%49 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %51 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xf32>
    %52 = linalg.fill ins(%cst : f32) outs(%51 : tensor<4x1x2xf32>) -> tensor<4x1x2xf32>
    %53 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31 : tensor<4x1x1xf32>) outs(%52 : tensor<4x1x2xf32>) attrs =  {iterator_ranges = [4, 1, 1], name = "concatenate"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<4x1x2xf32>
    %54 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xf32>
    %55 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53, %22 : tensor<4x1x2xf32>, tensor<4x1x2xf32>) outs(%54 : tensor<4x1x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x2xf32>
    %56 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xi32>
    %57 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%55 : tensor<4x1x2xf32>) outs(%56 : tensor<4x1x2xi32>) {
    ^bb0(%arg2: f32, %arg3: i32):
      %107 = arith.fptosi %arg2 : f32 to i32
      linalg.yield %107 : i32
    } -> tensor<4x1x2xi32>
    %58 = tile.gather %6 %57 {OutOfBoundsMode = 0 : i64, axis = 0 : index, batchDims = 1 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 1 : i64, nearestMode = 0 : i64} : (tensor<4x7x6xf32>, tensor<4x1x2xi32>) -> tensor<4x1xf32>
    %59 = linalg.init_tensor [4, 1, 1] : tensor<4x1x1xf32>
    %60 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %11 : tensor<4x1x1xf32>, tensor<4x1x1xf32>) outs(%59 : tensor<4x1x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.subf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x1xf32>
    %61 = tensor.collapse_shape %60 [[0], [1, 2]] : tensor<4x1x1xf32> into tensor<4x1xf32>
    %62 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %63 = linalg.generic {indexing_maps = [#map6, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%29, %61 : tensor<1xf32>, tensor<4x1xf32>) outs(%62 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %64 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %65 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%63, %41 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%64 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %66 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %67 = linalg.fill ins(%cst : f32) outs(%66 : tensor<4x1xf32>) -> tensor<4x1xf32>
    %68 = linalg.generic {indexing_maps = [#map8, #map7], iterator_types = ["parallel", "parallel"]} ins(%65 : tensor<4x1xf32>) outs(%67 : tensor<4x1xf32>) attrs =  {iterator_ranges = [4, 1], name = "broadcast"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<4x1xf32>
    %69 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %70 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%58, %68 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%69 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %71 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %72 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%50, %70 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%71 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %73 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xf32>
    %74 = linalg.fill ins(%cst : f32) outs(%73 : tensor<4x1x2xf32>) -> tensor<4x1x2xf32>
    %75 = linalg.generic {indexing_maps = [#map1, #map5], iterator_types = ["parallel", "parallel", "window"]} ins(%38 : tensor<4x1x1xf32>) outs(%74 : tensor<4x1x2xf32>) attrs =  {iterator_ranges = [4, 1, 1], name = "concatenate"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<4x1x2xf32>
    %76 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xf32>
    %77 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %75 : tensor<4x1x2xf32>, tensor<4x1x2xf32>) outs(%76 : tensor<4x1x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x2xf32>
    %78 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xi32>
    %79 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77 : tensor<4x1x2xf32>) outs(%78 : tensor<4x1x2xi32>) {
    ^bb0(%arg2: f32, %arg3: i32):
      %107 = arith.fptosi %arg2 : f32 to i32
      linalg.yield %107 : i32
    } -> tensor<4x1x2xi32>
    %80 = tile.gather %6 %79 {OutOfBoundsMode = 0 : i64, axis = 0 : index, batchDims = 1 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 1 : i64, nearestMode = 0 : i64} : (tensor<4x7x6xf32>, tensor<4x1x2xi32>) -> tensor<4x1xf32>
    %81 = linalg.init_tensor [4, 1, 1] : tensor<4x1x1xf32>
    %82 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %19 : tensor<4x1x1xf32>, tensor<4x1x1xf32>) outs(%81 : tensor<4x1x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.subf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x1xf32>
    %83 = tensor.collapse_shape %82 [[0], [1, 2]] : tensor<4x1x1xf32> into tensor<4x1xf32>
    %84 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %85 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%36, %83 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%84 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %86 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %87 = linalg.fill ins(%cst : f32) outs(%86 : tensor<4x1xf32>) -> tensor<4x1xf32>
    %88 = linalg.generic {indexing_maps = [#map8, #map7], iterator_types = ["parallel", "parallel"]} ins(%85 : tensor<4x1xf32>) outs(%87 : tensor<4x1xf32>) attrs =  {iterator_ranges = [4, 1], name = "broadcast"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<4x1xf32>
    %89 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %90 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%80, %88 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%89 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %91 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %92 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%72, %90 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%91 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %93 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xf32>
    %94 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53, %75 : tensor<4x1x2xf32>, tensor<4x1x2xf32>) outs(%93 : tensor<4x1x2xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1x2xf32>
    %95 = linalg.init_tensor [4, 1, 2] : tensor<4x1x2xi32>
    %96 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94 : tensor<4x1x2xf32>) outs(%95 : tensor<4x1x2xi32>) {
    ^bb0(%arg2: f32, %arg3: i32):
      %107 = arith.fptosi %arg2 : f32 to i32
      linalg.yield %107 : i32
    } -> tensor<4x1x2xi32>
    %97 = tile.gather %6 %96 {OutOfBoundsMode = 0 : i64, axis = 0 : index, batchDims = 1 : index, cubeCoeff = -7.500000e-01 : f64, interpolationMode = 3 : i64, mode = 1 : i64, nearestMode = 0 : i64} : (tensor<4x7x6xf32>, tensor<4x1x2xi32>) -> tensor<4x1xf32>
    %98 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %99 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%63, %83 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%98 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %100 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %101 = linalg.fill ins(%cst : f32) outs(%100 : tensor<4x1xf32>) -> tensor<4x1xf32>
    %102 = linalg.generic {indexing_maps = [#map8, #map7], iterator_types = ["parallel", "parallel"]} ins(%99 : tensor<4x1xf32>) outs(%101 : tensor<4x1xf32>) attrs =  {iterator_ranges = [4, 1], name = "broadcast"} {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<4x1xf32>
    %103 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %104 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%97, %102 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%103 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    %105 = linalg.init_tensor [4, 1] : tensor<4x1xf32>
    %106 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%92, %104 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%105 : tensor<4x1xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %107 = arith.addf %arg2, %arg3 : f32
      linalg.yield %107 : f32
    } -> tensor<4x1xf32>
    return %106 : tensor<4x1xf32>
  }
}
