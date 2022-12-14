import os
import argparse
import logging

from converter import Converter

def main():
    parser = argparse.ArgumentParser(prog="paddleconverter", description="Paddleconverter tool entry point")
    parser.add_argument("--in_dir", required=True, type=str, help='the input PyTorch file or directory.')
    parser.add_argument("--out_dir", required=True, type=str, help='the output Paddle directory.')
    parser.add_argument("--log_dir", default=None, type=str, help='the input PyTorch file or directory.')

    args = parser.parse_args()

    assert args.out_dir != args.in_dir, "--out_dir must be different from --in_dir"

    coverter = Converter(args.log_dir)
    coverter.run(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
