from workflow import *

# Read for debugging variable.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()
debug = args.debug

# Start debugging by listening on port.
if debug:
    import debugpy
    debugpy.listen(("localhost", 5678))
    debugpy.wait_for_client()
