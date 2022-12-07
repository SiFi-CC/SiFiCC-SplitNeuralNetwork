def parse(argcf):
    import os

    ###################################################
    # ConfigFileParser will look for the following settings
    #
    # # RootFile
    # # InputGenerator
    # # NeuralNetwork model
    # # TrainingStrategy
    # # Analysis scripts
    #
    # # Epochs
    # # Batch size
    # # Verbose

    dir_main = os.getcwd()

    # read config file
    config_file = argcf.read()
    list_config = config_file.split("\n")

    for i, row in enumerate(list_config):
        # skip rows with no entries
        if len(row) == 0:
            continue
        if row[0] == "#":
            # Test each config file input and determine their parameter
            if "Name of the origin root file" in row:
                # locate the origin root file
                # if the file is not found, the configfile parser will be stopped
                # if the corresponding npz file for meta-data found it will be generated
                param_rootfile = list_config[i+1]
                if not os.path.exists(dir_main + "/root_files/" + param_rootfile):
                    print("ERROR: Root file not found at ", dir_main + "/root_files/" + param_rootfile)
                    break
                else:
                    print("RootFile: ", param_rootfile)

                    # check if corresponding .npz file exists
                    if not os.path.exists(dir_main + "/npz_files/" + param_rootfile[:-5] + ".npz"):
                        print("ERROR: npz file not found at ", dir_main + "/root_files/" + param_rootfile[:-5] + ".npz")
                        print("Corresponding .npz file will be generated from ", param_rootfile)
                    else:
                        print("npz file: ", param_rootfile[:-5] + ".npz")

            if "Name of the neural network input file" in row:
                param_nninput_file = list_config[i+1]
                # TODO: continue here


            if "InputGenerator" in row:
                print("InputGenerator: ", list_config[i + 1])

            if "TensorflowModel" in row:
                print("TensforflowModel: ", list_config[i + 1])

            if "AnalysisModels" in row:
                print("AnalysisModels: ", list_config[i + 1])
