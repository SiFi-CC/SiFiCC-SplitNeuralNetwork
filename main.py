from classes import ArgParser
from classes import ConfigFileParser
from classes import NPZParser
from classes import SiFiCCNNTF
from classes import MetaData
from classes import RootParser

import sys
import os


########################################################################################################################

def main():
    # generate argument parser
    args = ArgParser.parser()

    # set main directory
    dir_main = os.getcwd()

    if args.cf is not None:
        print("Reading configfile ...")
        config_data = ConfigFileParser.parse(args.cf)

        if args.geninput:
            print("Generating input")
            # call RootParser with root file defined in config file
            root_data = RootParser(dir_main + "/root_files/" + config_data.root_file)

            # call InputGenerator method
            method_inputgenerator = __import__("inputgenerator." + config_data.input_generator, fromlist=[None])
            func_inputgenerator = getattr(method_inputgenerator, "gen_input")
            func_inputgenerator(root_data)

            # exit program
            sys.exit()

        DataCluster = NPZParser.parse(config_data.nninput)
        meta_data = MetaData.MetaData(config_data.metadata)

        # evaluate model expression
        model_method = __import__("models." + config_data.model, fromlist=[None])
        func1 = getattr(model_method, "return_model")
        model = func1(DataCluster.num_features())

        neuralnetwork = SiFiCCNNTF.SiFiCCNNTF(model=model,
                                              model_name=config_data.model,
                                              model_tag=config_data.modeltag)

        # evaluate training strategy expression
        trainingstrategy_method = __import__("trainingstrategy." + config_data.training_strategy, fromlist=[None])
        func2 = getattr(trainingstrategy_method, "train_strategy")
        func2(neuralnetwork, DataCluster)

        # evaluate analysis expression
        # analysis expressions can be a list of analysis methods
        for i in range(len(config_data.analysis)):
            analysis_method = __import__("analysis." + config_data.analysis[i], fromlist=[None])
            analysis = getattr(analysis_method, "analysis")
            analysis(neuralnetwork, DataCluster, meta_data)

    else:
        print("Error found in configfile!")
        print("CODE WILL BE ABORTED!")
        sys.exit()


if __name__ == "__main__":
    main()
