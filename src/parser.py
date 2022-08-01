class parser:
    def __int__(self):
        self.json_path
        self.mlir_input_dir
        self.mlir_output_dir
        self.test_dir
        self.mlir_build_dir
        self.level_zero_build

def init_parser():
    file = open("./config.txt", "r")
    for line in file:
        line = line.split("=")
        header = line[0].rstrip(" ")
        if header == "json_path":
            parser.json_path = line[1].strip(" \"\n")
        elif header == "mlir_input_dir":
            parser.mlir_input_dir = line[1].strip(" \"\n")
        elif header == "mlir_output_dir":
            parser.mlir_output_dir = line[1].strip(" \"\n")
        elif header == "test_dir":
            parser.test_dir = line[1].strip(" \"\n")
        elif header == "mlir_build_dir":
            parser.mlir_build_dir = line[1].strip(" \"\n")
        elif header == "level_zero_build":
            parser.level_zero_build = line[1].strip(" \"\n")
