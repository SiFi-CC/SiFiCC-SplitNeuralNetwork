class ConfigFileData:

    def __init__(self,
                 root_file,
                 npz_meta_file,
                 npz_nninput_file,
                 input_generator,
                 training_strategy,
                 analysis,
                 epochs,
                 verbose,
                 batch_size):

        self.root_file = root_file
        self.npz_meta_file = npz_meta_file
        self.npz_nninput_file = npz_nninput_file
        self.input_generator = input_generator
        self.training_strategy = training_strategy
        self.analysis = analysis

        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
