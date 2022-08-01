#map = affine_map<(d0, d1) -> (d0, d1)>
module @cast {
func.func @main() {
    %0= arith.constant dense<[[1, 2, 3], [4, 5, 4102], [16777223, 2147483656, 4294967295]]>:tensor<3x3xi64>
    %1 = call @test(%0) : (tensor<3x3xi64>) -> tensor<3x3xi32>
    %unranked = tensor.cast %1 : tensor<3x3xi32>to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> () 
    return 
} 
func.func private @printMemrefI32(tensor<*xi32>)
func.func @test(%arg0: tensor<3x3xi64>)->tensor<3x3xi32>{
    %0 = linalg.init_tensor [3, 3] : tensor<3x3xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<3x3xi64>) outs(%0 : tensor<3x3xi32>) {
    ^bb0(%arg1: i64, %arg2: i32):
      %2 = arith.trunci %arg1 : i64 to i32
      linalg.yield %2 : i32
    } -> tensor<3x3xi32>
    return %1 : tensor<3x3xi32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}} 
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 3] strides = {{.*}} data = 
// CHECK:   1
// CHECK:   2
// CHECK:   3
// CHECK:   4
// CHECK:   5
// CHECK:   4102
// CHECK:   16777223
// CHECK:   2147483656
// CHECK:   4294967295
