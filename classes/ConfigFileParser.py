def parse(argcf):
    import os

    from classes import ConfigData

    #######################################################
    # ConfigFileParser will look for the following settings
    #
    # # Name of the origin root file
    # # Method containing input generator
    # # Method containing Neural-Network model
    # # Method containing trainingstrategy
    # # Method containing Analysis models
    #
    # # Model nametag
    # # Generate input only
    # # load model
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

    param_modeltag = ""
    param_loadmodel = 0
    param_geninput = 0

    param_epochs = 10
    param_batchsize = 128
    param_verbose = 1

    ####################################################################################################################
    # config file readout

    # read config file and split config file string into list
    config_file = argcf.read()
    list_config = config_file.split("\n")

    for i, row in enumerate(list_config):
        # skip rows with no entries
        if len(row) == 0:
            continue

        if row[0] == "#":
            # Test each config file input and determine their parameter
            if "Name of the origin root file" in row:
                param_rootfile = list_config[i + 1]
                # break condition if file is not found
                if not os.path.exists(dir_main + "/root_files/" + param_rootfile):
                    print("ERROR: Root file not found at ", dir_main + "/root_files/" + param_rootfile)
                    return None

            if "Method containing input generator" in row:
                param_inputgenerator = list_config[i + 1]
                # break condition if file is not found
                if not os.path.exists(dir_main + "/inputgenerator/" + param_inputgenerator + ".py"):
                    print("ERROR: File not found at ",
                          dir_main + "/inputgenerator/" + param_inputgenerator + ".py")
                    return None

            if "Method containing Neural-Network model" in row:
                param_model = list_config[i + 1]
                # break condition if file is not found
                if not os.path.exists(dir_main + "/models/" + param_model + ".py"):
                    print("ERROR: File not found at ",
                          dir_main + "/models/" + param_model + ".py")
                    return None

            if "Method containing trainingstrategy" in row:
                param_trainingstrategy = list_config[i + 1]
                # break condition if file is not found
                if not os.path.exists(dir_main + "/trainingstrategy/" + param_trainingstrategy + ".py"):
                    print("ERROR: File not found at ",
                          dir_main + "/trainingstrategy/" + param_trainingstrategy + ".py")
                    return None

            if "Method containing Analysis models" in row:
                param_analysis = list_config[i + 1]
                # evaluate analysis parameter
                param_analysis = param_analysis.split(",")
                for j in range(len(param_analysis)):
                    param_analysis[j] = param_analysis[j].replace(" ", "")
                    if not os.path.exists(dir_main + "/analysis/" + param_analysis[j] + ".py"):
                        print("ERROR: File not found at ",
                              dir_main + "/analysis/" + param_analysis[j] + ".py")
                        continue


            # model name tag
            if "Model nametag" in row:
                param_modeltag = list_config[i + 1]

            # load model param (0: do not load model, 1: load model and test only)
            if "Load model" in row:
                param_loadmodel = int(list_config[i + 1])
                # TODO: check if value is valid, else use base value

            # param_geninput: (0: missing files will be generated and trained, 1: files will only be generated)
            if "generate neural network input only" in row:
                param_geninput = int(list_config[i + 1])
                # TODO: check if value is valid, else use base value

            # loose parameter
            if "No. of Epochs" in row:
                param_epochs = list_config[i + 1]
                # TODO: check if value is valid, else use base value

            # loose parameter
            if "batch size" in row:
                param_batchsize = list_config[i + 1]
                # TODO: check if value is valid, else use base value

            # loose parameter
            if "Verbose" in row:
                param_verbose = list_config[i + 1]
                # TODO: check if value is valid, else use base value

    ####################################################################################################################
    # parameter evaluation logic

    # check if meta data npz file corresponding to the given root file
    if not os.path.exists(dir_main + "/npz_files/" + param_rootfile[:-5] + ".npz"):
        print("Generating meta data file at: ", dir_main + "/root_files/" + param_rootfile)

        from classes.RootParser import RootParser
        root_data = RootParser(param_rootfile)
        root_data.export_npz(dir_main + "/npz_files/" + param_rootfile[:-5] + ".npz")

    # build configfile domain object
    config_data = ConfigData.ConfigData(root_file=param_rootfile,
                                        input_generator=param_inputgenerator,
                                        model=param_model,
                                        training_strategy=param_trainingstrategy,
                                        analysis=param_analysis,
                                        metadata=param_metafile,
                                        nninput=param_nninput,
                                        modeltag=param_modeltag,
                                        epochs=param_epochs,
                                        batch_size=param_batchsize,
                                        verbose=param_verbose)

    return config_data
