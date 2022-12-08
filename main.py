from classes import ArgParser
from classes import ConfigFileParser
from classes import NPZParser
from classes import SiFiCCNNTF

import models

args = ArgParser.parser()

if args.cf is not None:
    print("Config file found!")
    config_data = ConfigFileParser.parse(args.cf)
else:
    print("No configfile found!")

########################################################################################################################

DataCluster = NPZParser.parse(config_data.nninput)

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
    analysis(neuralnetwork, DataCluster)
