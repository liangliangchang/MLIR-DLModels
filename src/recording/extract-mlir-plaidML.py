import os

file = open("/home/lchang1/plaidml/plaidml/test-mlir-linalg-dump/edsl_test/list.txt","r")

cmd_pre = "/home/lchang1/plaidml/plaidml/build-x86_64/Release/bin/plaidml_edsl_tests_cc_test --gtest_filter="
header = ""
os.system("export PLAIDML_DUMP=/home/lchang1/plaidml/plaidml/dump/")
for line in file:
    if header == "":
        header = line.replace("\n", "")
    else:
        case = line.strip().replace("\n", "")
        test_case = header+case
        cmd = cmd_pre + test_case   
        t = os.system(cmd)
        print(cmd)
        mv = "mv /home/lchang1/plaidml/plaidml/dump/07_LowerTileToLinalg.mlir /home/lchang1/plaidml/plaidml/test-mlir-linalg-dump/edsl_test/test-case/"+test_case+".mlir"
        if t is not None:
            os.system(mv)


