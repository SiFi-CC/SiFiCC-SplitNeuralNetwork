# some header

def parser():
    """
    Defines argument parser for main file.

    return:
        parser
    """
    import argparse

    argparser = argparse.ArgumentParser(description="Some helper text", formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument("-cf", type=argparse.FileType('r'))

    return argparser.parse_args()
