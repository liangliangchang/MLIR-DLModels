import re
import numpy
import json
import sys

class inputVar:
    def __int__(self):
        self.arg = ""
        self.shape = ""
        self.value = ""

class MlirOut:
    def __int__(self):
        self.testCallArg = ""
        self.testCallShape = ""
        self.testRes = ""
        self.outTensorType = ""
        self.outShape = ""
        self.outType = ""
        self.outValue = ""
        self.inputVars = []


def getTypeStr(typeName):
    if typeName == "11":
        result = "i64"
    elif typeName == "12":
        result = "i64"
    elif typeName == "9":
        result = "i32"
    elif typeName == "10":
        result = "i32"
    elif typeName == "5":
        result = "i8"
    elif typeName == "6":
        result = "i8"
    elif typeName == "14":
        result = "f32"
    else:
        raise Exception("Unsupported Type",typeName)
    return result

def getPrintFunc(outType_):
    if outType_ == "i64":
        result = "printMemrefI64"
    elif outType_ == "i32":
        result = "printMemrefI32"
    elif outType_ == "i8":
        result = "printMemrefI32"
    elif outType_ == "i1":
        result = "printMemrefI32"
    elif outType_ == "f32":
        result = "printMemrefF32"
    elif outType_ == "f64":
        result = "printMemrefF64"
    else:
        raise Exception("Unsupported Type!", outType_)
    return result


# functionName->IO->Var
def parseJasonInput(json_, functionName_, mlirOut_):
    variables = json_[functionName_]["input"]
    inputID = 0
    for var in variables:
        ivar = inputVar()
        ivar.arg = "%" + str(inputID)
        inputID += 1
        mlirOut_.testCallArg += ivar.arg + ","
        values = var["value"].rstrip(",").split(",")
        shape = var["shape"].rstrip(",").split(",")
        dataType = getTypeStr(var["dataType"])
        # convert them to int
        if dataType == 'f32':
            values = [float(x) for x in values]
        else:
            values = [int(x) for x in values]
        shape = [int(x) for x in shape]
        # get value str
        values = numpy.array(values)
        value_list = values.reshape(shape).tolist()
        ivar.value = "%s" % value_list
        # get shape str
        shapeVar = ""

        for s in shape:
            shapeVar = shapeVar + str(s) + "x"
        ivar.shape = "tensor<" + shapeVar + dataType + ">"
        mlirOut_.testCallShape += ivar.shape + ","
        mlirOut_.inputVars.append(ivar)
    ouputInJson = json_[functionName_]["output"]

    mlirOut_.outValue = ouputInJson[0]["value"].rstrip(",").split(",")
    mlirOut_.outShape = ouputInJson[0]["shape"].rstrip(",").split(",")
    mlirOut_.outType = dataType
    mlirOut_.testRes = "%" + str(inputID)
    mlirOut_.testCallArg = mlirOut_.testCallArg.rstrip(",")
    mlirOut_.testCallShape = mlirOut_.testCallShape.rstrip(",")

mlir_path = sys.argv[1]
json_path = sys.argv[2]
out_mlir_path = sys.argv[3]
functionName = sys.argv[4]

# for debug
# mlir_path = "/data/lchang1/MLIR-DLModels/test/without-main/OpTest.BroadcastScalar.mlir"
# json_path = "/data/lchang1//MLIR-DLModels/src/data.json"
# out_mlir_path = "/data/lchang1/MLIR-DLModels/added-main/OpTest.BroadcastScalar.mlir"
# functionName = "OpTest.BroadcastScalar"

file = open(mlir_path)
jsonFile = open(json_path)
jsonInput = json.load(jsonFile)
outFile = open(out_mlir_path, "w")


mlirOut = MlirOut()
mlirOut.__int__()
parseJasonInput(jsonInput, functionName, mlirOut)
for line in file:
    # func @main(%arg0: tensor<128x1x1xf32>, %arg1: tensor<1x128x1xf32>, %arg2: tensor<1x1x128xf32>) -> tensor<f32>
    # writing inputs into the mlir file
    check_exact = re.compile(r'func @main(.*) -> (.*([if].*)>) {')
    check_exact_result = check_exact.search(line)
    if check_exact_result is not None:
        mlirOut.outTensorType = check_exact_result.group(2)
        mlirOut.outType = check_exact_result.group(3)
        outFile.write("func.func @main() {\n")
        for var in mlirOut.inputVars:
            outFile.write("    " + var.arg + "= arith.constant dense<" + var.value + ">:" + var.shape + "\n")
        outFile.write("    " +
                      mlirOut.testRes + " = call @test(" + mlirOut.testCallArg + ") : (" + mlirOut.testCallShape + ") "
                                                                                                                   "-> "
                                                                                                                   ""
                      + mlirOut.outTensorType + "\n")
        outFile.write(
            "    %unranked = tensor.cast " + mlirOut.testRes + " : " + mlirOut.outTensorType + "to tensor<*x" + mlirOut.outType + ">" + "\n")
        
        printFunc = getPrintFunc(mlirOut.outType)
        outFile.write("    call @" + printFunc + "(%unranked) : (tensor<*x" + mlirOut.outType + ">) -> () \n\
    return \n\
} \n\
func.func private @" + printFunc + "(tensor<*x" + mlirOut.outType + ">)\n")
        outFile.write("func.func @test" + check_exact_result.group(1) + "->" + check_exact_result.group(2) + "{\n")
    else:
        outFile.write(line)
# writing FileCheck flags into the mlir file

outFile.write("// CHECK: Unranked Memref base@ = {{0x[-9a-f]*}} \n")
outFile.write(("// CHECK-SAME: rank = {{.}} offset = {{.}} sizes = [")+", ".join(mlirOut.outShape) +("] strides = {{.*}} data = \n"))

for value in mlirOut.outValue:
    outFile.write("// CHECK:   " + value + "\n")

outFile.close()
