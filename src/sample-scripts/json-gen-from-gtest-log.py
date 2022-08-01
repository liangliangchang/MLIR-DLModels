import re
import json

class TensorVar:
    def __init__(self):
        self.index = 0
        self.value = ""
        self.shape = ""
        self.dataType = ""  # int or float


class CodeBlock:
    def __init__(self):
        self.inputs = []
        self.output = None  # only one output supported
        # track the value index
        self.shape_index = 0
        self.value_index = 0
        # track index to variable
        self.index_to_var = {}

    def codeBlockPrint(self):
        blockDic = {"input": [], "output": []}
        for var in self.inputs:
            varDic = {"index": var.index, "value": var.value,
                      "shape": var.shape, "dataType": var.dataType}
            blockDic["input"].append(varDic)

        var = self.output
        varDic = {"index": var.index, "value": var.value,
                  "shape": var.shape, "dataType": var.dataType}
        blockDic["output"].append(varDic)
        return blockDic


class Program:
    def __init__(self):
        self.program_name = ""
        self.blocks = []
        # track a undecided block
        self.undecided_block = None
        # track if a block will be added
        self.patternSet = set()
        self.programDic = {}

    def getProgramDic(self):
        for block_ in self.blocks:
            blockDic = block_.codeBlockPrint()
            function_name = block_.function_name
            self.programDic[function_name] = blockDic


def matched_re(inx_, result_, program_):
    if inx_ == 0:  # start_re
        code_block = CodeBlock()
        code_block.function_name = result_.group(1)
        program_.undecided_block = code_block
        program_.patternSet.clear()
        program_.patternSet.add(inx_)

    if inx_ == 1:  # dtype
        program_.patternSet.add(inx_)
        block = program_.undecided_block
        var = TensorVar()
        var.index = program_.undecided_block.shape_index
        var.dataType = result_.group(1)

        block.inputs.append(var)
        block.index_to_var[block.shape_index] = var
    if inx_ == 2:  # shape
        program_.patternSet.add(inx_)
        block = program_.undecided_block
        var = block.index_to_var[block.shape_index]
        var.shape = result_.group(1)
        block.shape_index += 1
    if inx_ == 3:  # output_dtype
        program_.patternSet.add(inx_)
        block = program_.undecided_block
        var = TensorVar()
        var.index = 0  # only one output supported
        var.dataType = result_.group(1)
        block.output = var
    if inx_ == 4:  # output_shape
        program_.patternSet.add(inx_)
        block = program_.undecided_block
        block.output.shape = result_.group(1)
    if inx_ == 5:  # input_value
        program_.patternSet.add(inx_)
        block = program_.undecided_block
        var = block.index_to_var[block.value_index]
        var.value = result_.group(1)
        block.value_index += 1
    if inx_ == 6:  # output_value
        program_.patternSet.add(inx_)
        block = program_.undecided_block
        block.output.value = result_.group(1)
    if inx_ == 7:  # end
        program_.patternSet.add(inx_)
        block = program_.undecided_block
        if program_.patternSet.__len__() == 8:
            program_.blocks.append(block)
        program_.undecided_block = None


start_re = re.compile(r'\[ RUN      \] (.*)')
input_dtype_re = re.compile(r'input_dtype:(.*)\n')
input_shape_re = re.compile(r'input_shape:(.*)\n')
output_dtype_re = re.compile(r'output_dtype:(.*)\n')
output_shape_re = re.compile(r'output_shape:(.*)\n')
input_value_re = re.compile(r'input:(.*)\n')
output_value_re = re.compile(r'output:(.*)\n')
end_re = re.compile(r'\[       OK \] (.*) \(.*')

expressions = [start_re, input_dtype_re, input_shape_re, output_dtype_re, output_shape_re, input_value_re,
               output_value_re, end_re]

file = open("LastTest.log", "r")
program = Program()

for line in file:
    for inx, exp in enumerate(expressions):
        result = exp.search(line)
        if result is not None:
            print("result" + str(inx) + " value " + result.group(1))
            matched_re(inx, result, program)
file.close()

program.getProgramDic()
a = program.programDic
with open('data-test.json', 'w') as f:
    json.dump(a, f, indent=2)
