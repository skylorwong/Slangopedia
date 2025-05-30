#!/usr/bin/env python

import argparse
import sys
import urban
import os

def get_args():
    parser = argparse.ArgumentParser(
        description='The Urban Dictionary slang application')
    parser.add_argument(
        'port',
        type=int,
        help='the port at which the server should listen')

    return parser.parse_args()

def main():
    # parse arguments
    args = get_args()

    try:
        urban.app.run(host='0.0.0.0', port=int(args.port), debug=True)
        #urban.app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)), debug=True)
    except Exception as ex:
        print(f"{sys.argv[0]}: {ex}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()