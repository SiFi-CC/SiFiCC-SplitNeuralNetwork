def parse(argcf):
    from src import ConfigData

    # define base settings for configfile input
    PARAM_ROOT_FILE_NAME = ""
    PARAM_INPUT_GENERATOR_NAME = ""
    PARAM_TRAINING_SCHEDULE_NAME = ""
    PARAM_MODEL_NAME_CLASSIFIER = ""
    PARAM_MODEL_NAME_REGRESSION1 = ""
    PARAM_MODEL_NAME_REGRESSION2 = ""
    PARAM_RUN_NAME = ""

    PARAM_LOAD_CLASSIFIER = 0
    PARAM_LOAD_REGRESSION1 = 0
    PARAM_LOAD_REGRESSION2 = 0

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
            PARAM_ROOT_FILE_NAME = str_row

        if "INPUT_GENERATOR_NAME:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("INPUT_GENERATOR_NAME:", "")
            PARAM_INPUT_GENERATOR_NAME = str_row

        if "TRAINING_SCHEDULE_NAME:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("TRAINING_SCHEDULE_NAME:", "")
            PARAM_TRAINING_SCHEDULE_NAME = str_row

        if "MODEL_NAME_CLASSIFIER:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("MODEL_NAME_CLASSIFIER:", "")
            PARAM_MODEL_NAME_CLASSIFIER = str_row

        if "MODEL_NAME_REGRESSION1:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("MODEL_NAME_REGRESSION1:", "")
            PARAM_MODEL_NAME_REGRESSION1 = str_row

        if "MODEL_NAME_REGRESSION2:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("MODEL_NAME_REGRESSION2:", "")
            PARAM_MODEL_NAME_REGRESSION2 = str_row

        if "RUN_NAME:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("RUN_NAME:", "")
            PARAM_RUN_NAME = str_row

        # additional settings
        if "LOAD_CLASSIFIER:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("LOAD_CLASSIFIER:", "")
            PARAM_LOAD_CLASSIFIER = bool(int(str_row))  # this feels wrong

        if "LOAD_REGRESSION1:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("LOAD_REGRESSION1:", "")
            PARAM_LOAD_REGRESSION1 = bool(int(str_row))  # this feels wrong

        if "LOAD_REGRESSION2:" in row:
            str_row = row.replace(" ", "")
            str_row = str_row.replace("LOAD_REGRESSION2:", "")
            PARAM_LOAD_REGRESSION2 = bool(int(str_row))  # this feels wrong


    #############################
    # parameter evaluation logic

    # print out all confirmed settings
    print("\n### config file settings found:")
    print("ROOT_FILE_NAME: {}".format(PARAM_ROOT_FILE_NAME))
    print("INPUT_GENERATOR_NAME: {}".format(PARAM_INPUT_GENERATOR_NAME))
    print("TRAINING_SCHEDULE_NAME: {}".format(PARAM_TRAINING_SCHEDULE_NAME))
    print("MODEL_NAME_CLASSIFIER: {}".format(PARAM_MODEL_NAME_CLASSIFIER))
    print("MODEL_NAME_REGRESSION1: {}".format(PARAM_MODEL_NAME_REGRESSION1))
    print("MODEL_NAME_REGRESSION2: {}".format(PARAM_MODEL_NAME_REGRESSION2))
    print("RUN_NAME: {}".format(PARAM_RUN_NAME))
    print("### Additional Settings")
    print("LOAD_CLASSIFIER: {}".format(PARAM_LOAD_CLASSIFIER))
    print("LOAD_REGRESSION1: {}".format(PARAM_LOAD_REGRESSION1))
    print("LOAD_REGRESSION2: {}".format(PARAM_LOAD_REGRESSION2))

    # build configfile domain object
    config_data = ConfigData.ConfigData(ROOT_FILE_NAME=PARAM_ROOT_FILE_NAME,
                                        INPUT_GENERATOR_NAME=PARAM_INPUT_GENERATOR_NAME,
                                        TRAINING_SCHEDULE_NAME=PARAM_TRAINING_SCHEDULE_NAME,
                                        MODEL_NAME_CLASSIFIER=PARAM_MODEL_NAME_CLASSIFIER,
                                        MODEL_NAME_REGRESSION1=PARAM_MODEL_NAME_REGRESSION1,
                                        MODEL_NAME_REGRESSION2=PARAM_MODEL_NAME_REGRESSION2,
                                        RUN_NAME=PARAM_RUN_NAME,
                                        LOAD_CLASSIFIER=PARAM_LOAD_CLASSIFIER,
                                        LOAD_REGRESSION1=PARAM_LOAD_REGRESSION1,
                                        LOAD_REGRESSION2=PARAM_LOAD_REGRESSION2
                                        )

    return config_data
