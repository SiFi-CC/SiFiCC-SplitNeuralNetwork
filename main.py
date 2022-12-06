from classes import ArgParser
from classes import ConfigFileParser

args = ArgParser.parser()

if args.cf is not None:
    print("Config file found!")
    ConfigFileParser.parse(args.cf)
else:
    print("No configfile found!")
