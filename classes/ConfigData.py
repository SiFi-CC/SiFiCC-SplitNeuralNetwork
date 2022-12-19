class ConfigData:

    def __init__(self,
                 ROOT_FILE_NAME,
                 INPUT_GENERATOR_NAME,
                 TRAINING_STRATEGY_NAME,
                 ANALYSIS_LIST,
                 MODEL_NAME,
                 RUN_TAG,
                 LOAD_MODEL,
                 NUMBER_EPOCHS
                 ):

        self.ROOT_FILE_NAME = ROOT_FILE_NAME
        self.INPUT_GENERATOR_NAME = INPUT_GENERATOR_NAME
        self.TRAINING_STRATEGY_NAME = TRAINING_STRATEGY_NAME
        self.ANALYSIS_LIST = ANALYSIS_LIST
        self.MODEL_NAME = MODEL_NAME
        self.RUN_TAG = RUN_TAG
        self.LOAD_MODEL = LOAD_MODEL
        self.NUMBER_EPOCHS = NUMBER_EPOCHS

        # generate additional needed settings from parameter given above
        self.META_DATA = self.ROOT_FILE_NAME[:-5] + ".npz"
        self.NN_INPUT = self.ROOT_FILE_NAME[:-5] + "_" + self.INPUT_GENERATOR_NAME + ".npz"

