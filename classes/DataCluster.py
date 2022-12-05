class DataCluster:

    def __init__(self, npz_file):
        self.npz_file = npz_file

        self.meta = npz_file["META"]
        self.cluster = npz_file["CLUSTER"]




    def x_train(self):
        # return training data
        return False
