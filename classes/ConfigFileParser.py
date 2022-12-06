def parse(argcf):
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

    # read config file
    config_file = argcf.read()

    for i, row in enumerate(config_file.split("\n")):
        if row[0] == "#":
            print(i, row, "| input: ", config_file.split("\n")[i+1])

