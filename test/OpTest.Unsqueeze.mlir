#map0 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module @unsqueeze {
func.func @main() {
    %0= arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]>:tensor<3x4xi64>
    %1 = call @test(%0) : (tensor<3x4xi64>) -> tensor<1x3x4xi64>
    %unranked = tensor.cast %1 : tensor<1x3x4xi64>to tensor<*xi64>
    call @printMemrefI64(%unranked) : (tensor<*xi64>) -> () 
    return 
} 
func.func private @printMemrefI64(tensor<*xi64>)
func.func @test(%arg0: tensor<3x4xi64>)->tensor<1x3x4xi64>{
    %c0_i64 = arith.constant 0 : i64
    %0 = linalg.init_tensor [1, 3, 4] : tensor<1x3x4xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x3x4xi64>) -> tensor<1x3x4xi64>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<3x4xi64>) outs(%1 : tensor<1x3x4xi64>) attrs =  {iterator_ranges = [1, 3, 4], name = "unsqueeze"} {
    ^bb0(%arg1: i64, %arg2: i64):
      linalg.yield %arg1 : i64
    } -> tensor<1x3x4xi64>
    return %2 : tensor<1x3x4xi64>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}} 
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [1, 3, 4] strides = {{.*}} data = 
// CHECK:   0
// CHECK:   1
// CHECK:   2
// CHECK:   3
// CHECK:   4
// CHECK:   5
// CHECK:   6
// CHECK:   7
// CHECK:   8
// CHECK:   9
// CHECK:   10
// CHECK:   11
