from classes import ArgParser

args = ArgParser.parser()

if args.cf is not None:
    print("Config file found!")
    with open(args.cf) as file:
        for row in file:
            print(row)
else:
    print("No configfile found!")
    