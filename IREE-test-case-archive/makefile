IREE_RUNNER = /home/lchang1/iree/iree/build-ll/tools/iree-run-mlir --iree-input-type=mhlo --iree-hal-target-backends=dylib-llvm-aot

IREE_INPUT = --function-input="1x224x224x3xf32"

MHLO_OPT = /home/lchang1/mhlo/mlir-hlo/build-ll/bin/mlir-hlo-opt -mhlo-test-unfuse-batch-norm -cse -hlo-legalize-to-linalg

MLIR_OPT = /home/lchang1/llvm/llvm-project/build-ll/bin/mlir-opt

MLIR_OPT_FLAGS = -convert-tensor-to-linalg -linalg-bufferize -arith-bufferize  -tensor-bufferize -func-bufferize -buffer-deallocation -convert-linalg-to-loops -convert-scf-to-cf  -convert-linalg-to-llvm -lower-affine -convert-scf-to-cf --convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts --arith-expand --convert-arith-to-llvm --convert-math-to-llvm -finalizing-bufferize

MLIR_CPU_RUNNER = /home/lchang1/llvm/llvm-project/build-ll/bin/mlir-cpu-runner -e main -entry-point-result=void    -shared-libs=/home/lchang1/llvm/llvm-project/build-ll/lib/libmlir_c_runner_utils.so,/home/lchang1/llvm/llvm-project/build-ll/lib/libmlir_runner_utils.so

CURRENT_PATH = /home/lchang1/mhlo/mlir-hlo/Liangliang-test

APP_PATH = $(CURRENT_PATH)/Usable-test-case/original-mhlo-dialect

APP_NAME = mobilenetv3-mhlo

run-iree:
	echo "----------- IREE runtime verify -----------"
	$(IREE_RUNNER) $(APP_PATH)/$(APP_NAME).mlir $(IREE_INPUT)

lower-to-linalg:
	echo "----------- lower MHLO to linalg -----------"
	$(MHLO_OPT) $(APP_PATH)/$(APP_NAME).mlir -o $(CURRENT_PATH)/$(APP_NAME)-linalg.mlir
	

add-main: lower-to-linalg
	echo "----------- you need to add the main function to $(APP_NAME)-linalg.mlir and rename it to $(APP_NAME)-linalg-addedMain.mlir  -----------"

lower-to-llvm: add-main
	echo "----------- lower linalg to llvm -----------"
	$(MLIR_OPT) $(CURRENT_PATH)/$(APP_NAME)-linalg-addedMain.mlir $(MLIR_OPT_FLAGS) -o $(CURRENT_PATH)/$(APP_NAME)-llvm.mlir


run-llvm: lower-to-llvm
	$(MLIR_CPU_RUNNER) $(CURRENT_PATH)/$(APP_NAME)-llvm.mlir

clean:
	-rm *.mlir
