def config_verify(ConfigData):
    """
    Take ConfigData object and verify is all needed files are available

    Args:
        ConfigData:

    Returns:

    """
    import os

    # define parent directory
    dir_main = os.getcwd()
    dir_root = dir_main + "/root_files/"
    dir_npz = dir_main + "/npz_files/"
    dir_models = dir_main + "/models/"
    dir_inputgenerator = dir_main + "/inputgenerator/"
    dir_trainingstrategy = dir_main + "/trainingstrategy/"
    dir_analysis = dir_main + "/analysis/"

    # Base setting True, if one condition is not met, False will be returned
    returner = True

    print("\nVerifying config file settings...")
    # check if root file exists
    if not os.path.exists(dir_root + ConfigData.ROOT_FILE_NAME):
        print("ERROR: Root file not found at ", dir_root + ConfigData.ROOT_FILE_NAME)
        return False
    print("Root file found at ", dir_root + ConfigData.ROOT_FILE_NAME)

    # check if meta data npz file corresponding to the given root file exists
    if not os.path.exists(dir_npz + ConfigData.META_DATA):
        print("Generating meta data file at: ", dir_npz + ConfigData.META_DATA)

        from classes.RootParser import RootParser
        root_parser = RootParser(dir_root + ConfigData.ROOT_FILE_NAME)
        root_parser.export_npz(dir_npz + ConfigData.META_DATA)
    print("Meta data file found at: ", dir_npz + ConfigData.META_DATA)

    if not os.path.exists(dir_inputgenerator + "InputGenerator" + ConfigData.INPUT_GENERATOR_NAME + ".py"):
        print("ERROR: InputGenerator not found at ",
              dir_inputgenerator + "InputGenerator" + ConfigData.INPUT_GENERATOR_NAME + ".py")
        return False
    print("InputGenerator found at ", dir_inputgenerator + "InputGenerator" + ConfigData.INPUT_GENERATOR_NAME + ".py")

    if not os.path.exists(dir_trainingstrategy + "TrainingStrategy" + ConfigData.TRAINING_STRATEGY_NAME + ".py"):
        print("ERROR: TrainingStrategy not found at ",
              dir_trainingstrategy + "TrainingStrategy" + ConfigData.TRAINING_STRATEGY_NAME + ".py")
        return False
    print("TrainingStrategy found at ",
          dir_trainingstrategy + "TrainingStrategy" + ConfigData.TRAINING_STRATEGY_NAME + ".py")

    for i in range(len(ConfigData.ANALYSIS_LIST)):
        if not os.path.exists(dir_analysis + "Analysis" + ConfigData.ANALYSIS_LIST[i] + ".py"):
            print("ERROR: Analysis method not found at ",
                  dir_analysis + "Analysis" + ConfigData.ANALYSIS_LIST[i] + ".py")
            return False
        print("Analysis method found at ", dir_analysis + "Analysis" + ConfigData.ANALYSIS_LIST[i] + ".py")

    if not os.path.exists(dir_models + "Model" + ConfigData.MODEL_NAME + ".py"):
        print("ERROR: Model not found at ",
              dir_models + "Model" + ConfigData.MODEL_NAME + ".py")
        return False
    print("Model found at ", dir_models + "Model" + ConfigData.MODEL_NAME + ".py")

    # TODO: generation of training set if not exist
    # TODO: verification of results folder
    # TODO: verification of loaded model

    return True
