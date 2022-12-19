def parse(argcf):
    import os

    from classes import ConfigData

    #######################################################
    # ConfigFileParser will look for the following settings
    #
    # ROOT_FILE_NAME
    # INPUT_GENERATOR_NAME
    # TRAINING_STRATEGY_NAME
    # ANALYSIS_LIST
    # MODEL_NAME
    # RUN_TAG
    #
    # LOAD_MODEL
    #######################################################

    # define base settings for configfile input
    param_ROOT_FILE_NAME = ""
    param_INPUT_GENERATOR_NAME = ""
    param_TRAINING_STRATEGY_NAME = ""
    param_ANALYSIS_LIST = []
    param_MODEL_NAME = ""
    param_RUN_TAG = ""

    param_LOAD_MODEL = 0
    param_NUMBER_EPOCHS = 5

    #####################
    # config file readout

    # read config file and split config file string into list
    config_file = argcf.read()
    list_config = config_file.split("\n")

    for i, row in enumerate(list_config):
        # skip rows with no entries
        if len(row) == 0:
            continue
        # skip rows starting with "#"
        if row[0] == "#":
            continue

        # check row content for all config file settings
        if "ROOT_FILE_NAME:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("ROOT_FILE_NAME:", "")
            param_ROOT_FILE_NAME = str_row

        if "INPUT_GENERATOR_NAME:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("INPUT_GENERATOR_NAME:", "")
            param_INPUT_GENERATOR_NAME = str_row

        if "TRAINING_STRATEGY_NAME:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("TRAINING_STRATEGY_NAME:", "")
            param_TRAINING_STRATEGY_NAME = str_row

        if "ANALYSIS_LIST:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("ANALYSIS_LIST:", "")
            param_ANALYSIS_LIST = str_row.split(",")

        if "MODEL_NAME:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("MODEL_NAME:", "")
            param_MODEL_NAME = str_row

        if "RUN_TAG:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("RUN_TAG:", "")
            param_RUN_TAG = str_row

        # additional settings
        if "LOAD_MODEL:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("LOAD_MODEL:", "")
            param_LOAD_MODEL = bool(int(str_row))  # this feels wrong

        # additional settings
        if "NUMBER_EPOCHS:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("NUMBER_EPOCHS:", "")
            param_NUMBER_EPOCHS = int(str_row)

    #############################
    # parameter evaluation logic

    # print out all confirmed settings
    print("\n### config file settings found:")
    print("ROOT_FILE_NAME: {}".format(param_ROOT_FILE_NAME))
    print("INPUT_GENERATOR_NAME: {}".format(param_INPUT_GENERATOR_NAME))
    print("TRAINING_STRATEGY_NAME: {}".format(param_TRAINING_STRATEGY_NAME))
    print("ANALYSIS_LIST: {}".format(param_ANALYSIS_LIST))
    print("MODEL_NAME: {}".format(param_MODEL_NAME))
    print("RUN_TAG: {}".format(param_RUN_TAG))
    print("### Additional Settings")
    print("LOAD_MODEL: {}".format(param_LOAD_MODEL))
    print("NUMBER_EPOCHS: {}".format(param_NUMBER_EPOCHS))

    # build configfile domain object
    config_data = ConfigData.ConfigData(ROOT_FILE_NAME=param_ROOT_FILE_NAME,
                                        INPUT_GENERATOR_NAME=param_INPUT_GENERATOR_NAME,
                                        TRAINING_STRATEGY_NAME=param_TRAINING_STRATEGY_NAME,
                                        ANALYSIS_LIST=param_ANALYSIS_LIST,
                                        MODEL_NAME=param_MODEL_NAME,
                                        RUN_TAG=param_RUN_TAG,
                                        LOAD_MODEL=param_LOAD_MODEL,
                                        NUMBER_EPOCHS=param_NUMBER_EPOCHS
                                        )

    return config_data
