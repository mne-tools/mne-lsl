import glob
import os
import sys
from importlib import import_module
from pathlib import Path

import bsl


def run():
    """Entrypoint for bsl <command> usage."""
    bsl_root = Path(__file__).parent.parent
    valid_commands = sorted(glob.glob(str(bsl_root / "commands" / "bsl_*.py")))
    valid_commands = [file.split(os.path.sep)[-1][4:-3] for file in valid_commands]

    def print_help():
        print("Usage: BrainStreamingLayer command options\n")
        print("Accepted commands:\n")
        for command in valid_commands:
            print("\t- %s" % command)
        print('\nExample: bsl stream_player StreamPlayer "path to .fif file"')

    if len(sys.argv) == 1 or "help" in sys.argv[1] or "-h" in sys.argv[1]:
        print_help()
    elif sys.argv[1] == "--version":
        print("BrainStreamingLayer %s" % bsl.__version__)
    elif sys.argv[1] not in valid_commands:
        print('Invalid command: "%s"\n' % sys.argv[1])
        print_help()
    else:
        cmd = sys.argv[1]
        cmd = import_module(".bsl_%s" % (cmd,), "bsl.commands")
        sys.argv = sys.argv[1:]
        cmd.run()
