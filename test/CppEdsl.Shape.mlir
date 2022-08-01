module @shape {
func.func @main() {
    %0= arith.constant dense<[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]>:tensor<2x3xf32>
    %1 = call @test(%0) : (tensor<2x3xf32>) -> tensor<2xi32>
    %unranked = tensor.cast %1 : tensor<2xi32>to tensor<*xi32>
    call @printMemrefI32(%unranked) : (tensor<*xi32>) -> () 
    return 
} 
func.func private @printMemrefI32(tensor<*xi32>)
func.func @test(%arg0: tensor<2x3xf32>)->tensor<2xi32>{
    %cst = arith.constant dense<[2, 3]> : tensor<2xi32>
    return %cst : tensor<2xi32>
  }
}
// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}} 
// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [2] strides = {{.*}} data = 
// CHECK:   2
// CHECK:   3
