class ConfigData:

    def __init__(self,
                 ROOT_FILE_NAME,
                 INPUT_GENERATOR_NAME,
                 TRAINING_SCHEDULE_NAME,
                 MODEL_NAME_CLASSIFIER,
                 MODEL_NAME_REGRESSION1,
                 MODEL_NAME_REGRESSION2,
                 RUN_NAME,
                 LOAD_CLASSIFIER,
                 LOAD_REGRESSION1,
                 LOAD_REGRESSION2,
                 ):

        self.ROOT_FILE_NAME = ROOT_FILE_NAME
        self.INPUT_GENERATOR_NAME = INPUT_GENERATOR_NAME
        self.TRAINING_SCHEDULE_NAME = TRAINING_SCHEDULE_NAME
        self.MODEL_NAME_CLASSIFIER = MODEL_NAME_CLASSIFIER
        self.MODEL_NAME_REGRESSION1 = MODEL_NAME_REGRESSION1
        self.MODEL_NAME_REGRESSION2 = MODEL_NAME_REGRESSION2
        self.RUN_NAME = RUN_NAME

        self.LOAD_CLASSIFIER = LOAD_CLASSIFIER
        self.LOAD_REGRESSION1 = LOAD_REGRESSION1
        self.LOAD_REGRESSION2 = LOAD_REGRESSION2

        # generate additional needed settings from parameter given above
        self.META_DATA = self.ROOT_FILE_NAME[:-5] + ".npz"
        self.NN_INPUT = self.ROOT_FILE_NAME[:-5] + "_" + self.INPUT_GENERATOR_NAME + ".npz"

