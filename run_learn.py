import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "gen_script_location",
    help="Location of ..."
)
args = parser.parse_args()

load_generator(args.gen_script_location)
# model
# gateset
