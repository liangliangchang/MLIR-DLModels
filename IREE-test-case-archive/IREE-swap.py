import re


def print_used_mhlo_op(file):
    for line in file:
        # cut the line into pieces
        # splited_line = line.strip().split()
        # for word in splited_line:

        # learn about regex
        # https://www.rexegg.com/regex-quickstart.html
        # https://regexper.com/#mhlo.*%5Cs%2B
        # note: add what should be there not, not exclude them
        pattern_mhlo = re.compile(r'"mhlo.*"|mhlo\.\w+')
        match_result = pattern_mhlo.search(line)
        if match_result is not None:
            print(match_result.group(0))


file = open("./original_iree_model/mobilenetv3_fake_weights.mlir", "r")
# print_used_mhlo_op(file)

result = []
global_to_const = {}
addr_to_const = {}
#
for line in file:
    pattern_mhlo = re.compile(r'util\.global private @"(.*)" {noinline} = (.*)')
    match_result = pattern_mhlo.search(line)
    if match_result is not None:
        # print("global_name:"+ match_result.group(1))
        # print("constant_for_arith:"+ match_result.group(2))
        global_to_const[match_result.group(1)] = match_result.group(2)

    # %0 = util.global.address @"__iree_flow___sm_node188__m.layer-2.kernel" : !util.ptr<tensor<7x7x3x64xf32>>
    pattern_mhlo = re.compile(r'(%.*) = util\.global\.address @"(.*)"')
    match_result = pattern_mhlo.search(line)
    if match_result is not None:
        # print("address_var:"+ match_result.group(1))
        # print("global_name:"+ match_result.group(2))
        addr_to_const[match_result.group(1)] = global_to_const[match_result.group(2)]

    # %332 = util.global.load.indirect %5
    pattern_mhlo = re.compile(r'(%.*) = util\.global\.load\.indirect (%.*) :')
    match_result = pattern_mhlo.search(line)
    if match_result is not None:
        print("result_var:" + match_result.group(1))
        print("constant_for_arith:" + addr_to_const[match_result.group(2)])
        result.append(match_result.group(1) + " = arith.constant " + addr_to_const[match_result.group(2)])

file.close()

file = open("./result.mlir", "w")
for line in result:
    file.write(line)
    file.write("\n")
file.close()
