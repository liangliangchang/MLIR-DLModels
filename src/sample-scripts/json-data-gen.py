import re
import json
import sys

class TensorVar:
    def __init__(self):
        self.typeName = ""
        self.valueName = ""
        self.index = 0
        self.value = ""
        self.shape = ""
        self.dataType = ""  # int or float
        self.io = ""  # input or output


class UndecidedVar:
    def __init__(self):
        self.valueName = ""
        self.value = {}


class CodeBlock:
    def __init__(self):
        self.function_name = ""
        self.inputs = []
        self.outputs = []
        self.undecidedVars = []
        self.nameToTypeVar = {}
        self.indexToVar = {}
        self.outputIndexToVar = {}
        self.nameToUndecidedVar = {}

    def updateValueShape(self, idx_, value_name_):
        self.indexToVar[idx_].valueName = value_name_
        self.indexToVar[idx_].value = self.nameToUndecidedVar[value_name_].value

    def updateOutputValueShape(self, idx_, value_name_):
        self.outputIndexToVar[idx_].valueName = value_name_
        self.outputIndexToVar[idx_].value = self.nameToUndecidedVar[value_name_].value

    def codeBlockPrint(self):
        blockDic = {"input": [], "output": []}
        for var in self.inputs:
            varDic = {"typeName": var.typeName, "valueName": var.valueName, "index": var.index, "value": var.value,
                      "shape": var.shape, "dataType": var.dataType}
            blockDic["input"].append(varDic)

        for var in self.outputs:
            varDic = {"typeName": var.typeName, "valueName": var.valueName, "index": var.index, "value": var.value,
                      "shape": var.shape, "dataType": var.dataType}
            blockDic["output"].append(varDic)
        return blockDic


class Program:
    def __init__(self):
        self.program_name = ""
        self.blocks = []
        self.currentBID = 0
        self.programDic = {}
        self.functionNameToBlock = {}

    def getProgramDic(self):
        for block_ in self.blocks:
            blockDic = block_.codeBlockPrint()
            function_name = block_.function_name
            self.programDic[function_name] = blockDic



def matched_re(inx_, result_, program_):
    if inx_ == 0:  # test_f
        code_block = CodeBlock()
        code_block.function_name = result_.group(2)
        program_.blocks.append(code_block)
        program_.currentBID = program_.currentBID + 1
        program_.functionNameToBlock[code_block.function_name] = code_block
    if inx_ == 1:  # shape
        var = TensorVar()
        var.typeName = result_.group(1)
        var.dataType = result_.group(2)
        var.shape = result_.group(3)
        program_.blocks[program_.currentBID - 1].inputs.append(var)
        program_.blocks[program_.currentBID - 1].nameToTypeVar[var.typeName] = var
    if inx_ == 2:  # make_program
        program_.program_name = result_.group(1)
        for idx, name_mk in enumerate(result_.group(2).split(", ")):
            program_.blocks[program_.currentBID - 1].nameToTypeVar[name_mk].index = idx
            program_.blocks[program_.currentBID - 1].nameToTypeVar[name_mk].io = "input"
            program_.blocks[program_.currentBID - 1].indexToVar[idx] = \
                program_.blocks[program_.currentBID - 1].nameToTypeVar[name_mk]

        # output
        var = TensorVar()  # index, shape, type, IO
        var.index = 0
        var.io = "output"
        var.shape = result_.group(3)
        program_.blocks[program_.currentBID - 1].outputs.append(var)
        program_.blocks[program_.currentBID - 1].outputIndexToVar[0] = var
    if inx_ == 3:  # value
        var = UndecidedVar()
        var.valueName = result_.group(1)
        var.value = result_.group(2)
        program_.blocks[program_.currentBID - 1].undecidedVars.append(var)
        program_.blocks[program_.currentBID - 1].nameToUndecidedVar[var.valueName] = var
    if inx_ == 4:  # check_exact4
        for idx, value_name in enumerate(result_.group(1).split(", ")):
            program_.blocks[program_.currentBID - 1].updateValueShape(idx, value_name)
        for idx, value_name in enumerate(result_.group(2).split(", ")):
            program_.blocks[program_.currentBID - 1].updateOutputValueShape(idx, value_name)


test_f_re = re.compile(r'TEST_F\((.*), (.*)\)')
shape_re = re.compile(r' +auto (.*) = Placeholder\(DType::(.*), {(.*)}\)')
make_program_re = re.compile(r'makeProgram\("(.*)", {(.*)}, {(.*)}\)')
value_re = re.compile(r'std::vector<.*> (.*){(.*)}')
check_exact_re = re.compile(r'checkExact\(program, {(.*)}, {(.*)}\)')

expressions = [test_f_re, shape_re, make_program_re, value_re, check_exact_re]



# py (input gtest) (output json)
file = open(sys.argv[1], "r")
program = Program()

for line in file:
    for inx, exp in enumerate(expressions):
        result = exp.search(line)
        if result is not None:
            matched_re(inx, result, program)
file.close()

program.getProgramDic()
a = program.programDic
with open(sys.argv[2], 'w') as f:
    json.dump(a, f, indent=2)

