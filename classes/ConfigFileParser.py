def parse(argcf):
    import os

    from classes import ConfigData

    #######################################################
    # ConfigFileParser will look for the following settings
    #
    # # RootFile
    # # inputgenerator
    # # NeuralNetwork model
    # # trainingstrategy
    # # Analysis scripts
    #
    # # Epochs
    # # Batch size
    # # Verbose
    #
    #######################################################

    # define parent directory
    dir_main = os.getcwd()

    # define base settings for configfile input
    param_rootfile = dir_main + "/root_files/" + "OptimisedGeometry_BP0mm_2e10protons.root"
    param_metafile = dir_main + "/npz_files/" + "OptimisedGeometry_BP0mm_2e10protons.npz"
    param_nninput = dir_main + "/npz_files/" + "NNInputDenseBase_OptimisedGeometry_BP0mm_2e10protons.npz"
    param_inputgenerator = "InputGeneratorDenseBase"
    param_model = "ModelDenseBase"
    param_trainingstrategy = "TrainingStrategyDenseBase"
    param_analysis = "AnalysisMetrics"

    param_epochs = 10
    param_batchsize = 128
    param_verbose = 1

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
                param_rootfile = list_config[i + 1]
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

            if "Method containing input generator" in row:
                param_inputgenerator = list_config[i + 1]
                # TODO: logic

            if "Method containing Neural-Network model" in row:
                param_model = list_config[i + 1]
                # TODO: logic

            if "Method containing trainingstrategy" in row:
                param_trainingstrategy = list_config[i + 1]
                # TODO: logic

            if "Method containing Analysis models" in row:
                param_analysis = list_config[i + 1]
                # TODO: add support for multiple inputs in form of list

            # loose parameter
            if "No. of Epochs" in row:
                param_epochs = list_config[i + 1]
                # TODO: logic

            # loose parameter
            if "batch size" in row:
                param_batchsize = list_config[i + 1]
                # TODO: logic

            # loose parameter
            if "Verbose" in row:
                param_verbose = list_config[i + 1]
                # TODO: logic

    # build configfile domain object
    config_data = ConfigData.ConfigData(root_file=param_rootfile,
                                        input_generator=param_inputgenerator,
                                        model=param_model,
                                        training_strategy=param_trainingstrategy,
                                        analysis=param_analysis,
                                        metadata=param_metafile,
                                        nninput=param_nninput,
                                        epochs=param_epochs,
                                        batch_size=param_batchsize,
                                        verbose=param_verbose)

    return config_data
