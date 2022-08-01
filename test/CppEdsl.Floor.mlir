#map = affine_map<(d0, d1) -> (d0, d1)>
module @floor {
func.func @main() {
    %0= arith.constant dense<[[1.1, 9.21, 3.0], [-0.4, -7.0, 0.6], [1.0, 0.0, -6.6]]>:tensor<3x3xf32>
    %1 = call @test(%0) : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %unranked = tensor.cast %1 : tensor<3x3xf32>to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> () 
    return 
} 
func.func private @printMemrefF32(tensor<*xf32>)
func.func @test(%arg0: tensor<3x3xf32>)->tensor<3x3xf32>{
    %0 = linalg.init_tensor [3, 3] : tensor<3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<3x3xf32>) outs(%0 : tensor<3x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = math.floor %arg1 : f32
      linalg.yield %2 : f32
    } -> tensor<3x3xf32>
    return %1 : tensor<3x3xf32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}} 
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [3, 3] strides = {{.*}} data = 
// CHECK:   1
// CHECK:   9
// CHECK:   3
// CHECK:   -1
// CHECK:   -7
// CHECK:   0
// CHECK:   1
// CHECK:   0
// CHECK:   -7
