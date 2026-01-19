import sys
from argparse import ArgumentError
from typing import Dict, Any


def parse_args() -> Dict[str, Any]:
    parsed_args = {}
    for idx, arg in enumerate(sys.argv[1:], 1):
        if '=' in arg:
            terms = arg.split('=')
            if len(terms) != 2:
                raise ArgumentError("The argument given \"%s\" is not valid." % (arg,))
            attr = terms[0]
            val = terms[1]
            if not attr:
                raise ArgumentError("The argument given \"%s\" is not valid." % (arg,))
            if not val:
                raise ArgumentError("The argument given \"%s\" is not valid." % (arg,))
            parsed_args[attr] = val
        else:
            parsed_args[str(idx)] = arg
    return parsed_args
