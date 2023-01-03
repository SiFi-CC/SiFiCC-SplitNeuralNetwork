import os


def setup_directory():
    # define directory paths
    dir_main = os.getcwd()

    # check if needed directories exist, else create them
    if not os.path.isdir(dir_main + "/root_files/"):
        os.mkdir(dir_main + "/root_files/")

    if not os.path.isdir(dir_main + "/npz_files/"):
        os.mkdir(dir_main + "/npz_files/")

    if not os.path.isdir(dir_main + "/models/"):
        os.mkdir(dir_main + "/models/")

    if not os.path.isdir(dir_main + "/results/"):
        os.mkdir(dir_main + "/results/")

