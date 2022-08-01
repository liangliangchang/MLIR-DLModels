from os import listdir
from os.path import isfile, join
from os import system
import parser

parser.init_parser()  
test_dir = parser.parser.test_dir
mlir_build_dir = parser.parser.mlir_build_dir
level_zero_build = parser.parser.level_zero_build

# for debug
# test_dir = "/data/lchang1/MLIR-DLModels/test/"
# mlir_build_dir = "/data/lchang1/mlir-llvm/"
# level_zero_build = "/data/lchang1/mlir-extensions/build/"

FileCheck = mlir_build_dir+"bin/FileCheck"
level_zero_runner = level_zero_build+ "mlir/tools/level_zero_runner/level_zero_runner"
other_flags = "--entry-point-result=void --shared-libs="+ mlir_build_dir +"lib/libmlir_runner_utils.so,"+level_zero_build+"dpcomp_gpu_runtime/libdpcomp-gpu-runtime.so,"+level_zero_build+"dpcomp_runtime/libdpcomp-runtime.so" +" | "+ FileCheck +" "

test_files = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
for test in test_files:
    cmd = level_zero_runner+" " +test_dir+test+" "+ other_flags +test_dir+test
    print(cmd)
    t = system(cmd)
    print(t)

    
