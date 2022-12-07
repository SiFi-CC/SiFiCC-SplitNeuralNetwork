class ConfigData:

    def __init__(self,
                 root_file,
                 input_generator,
                 training_strategy,
                 analysis,
                 metadata,
                 nninput,
                 model,
                 epochs,
                 verbose,
                 batch_size):

        self.root_file = root_file
        self.input_generator = input_generator
        self.training_strategy = training_strategy
        self.model = model
        self.analysis = analysis
        self.metadata = metadata
        self.nninput = nninput

        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
