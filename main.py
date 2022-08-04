import argparse
import sys
from absl import app
from absl.flags import argparse_flags

import train
import test_autoreg as test
from benchmark import Benchmark


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # High-level options.
    parser.add_argument("--name", type=str, help="Name of experiment.")
    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Report bitrate and distortion when training or compressing.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory or file containing weights to reload before training and "
        "Experiment directory containing params.json",
    )
    parser.add_argument(
        "--use_adversarial_loss",
        help="Use GAN loss for training.",
        type=bool,
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
        "to train) a new model. 'compress' reads an image file (lossless "
        "PNG format) and writes a compressed binary file. 'decompress' "
        "reads a binary file and reconstructs the image (in PNG format). "
        "input and output filenames need to be provided for the latter "
        "two options. Invoke '<command> -h' for more information.",
    )

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model.",
    )
    train_cmd.add_argument(
        "--train-data-dir",
        default="data/train",
        help="Directory containing the training dataset",
    )
    train_cmd.add_argument(
        "--eval-data-dir",
        default="data/benchmark/kodak",
        help="Directory containing the evaluation/validation dataset",
    )
    train_cmd.add_argument(
        "--batchsize", help="Training mini-batch size", default=4, type=int
    )
    # Experiment arguments
    train_cmd.add_argument(
        "--num-parallel-calls",
        help="Number of cores for multithreading workloads",
        default=4,
        type=int,
    )
    train_cmd.add_argument(
        "--epochs", help="Number of epochs to run training", default=1000, type=int
    )
    train_cmd.add_argument(
        "--save-summary-steps",
        help="Number of steps to save summary",
        default=100,
        type=int,
    )
    train_cmd.add_argument(
        "--random-seed", help="Random seed for TensorFlow", default=None, type=int
    )
    # Performance tuning parameters
    train_cmd.add_argument(
        "--allow-growth",
        help="Whether to enable allow_growth in GPU_Options",
        default=True,
        type=bool,
    )
    train_cmd.add_argument(
        "--xla",
        help="Whether to enable XLA auto-jit compilation",
        default=False,
        type=bool,
    )
    train_cmd.add_argument(
        "--save-profiling-steps",
        help="Number of steps to save profiling",
        default=0,
        type=int,
    )
    # Argument to turn on all logging
    train_cmd.add_argument(
        "--log-verbosity",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
        default="INFO",
        help="Set logging verbosity",
    )

    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.",
    )

    # 'decompress' subcommand.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file, reconstructs the image, and writes back "
        "a PNG file.",
    )
    # 'benchmark' subcommand.
    benchmark_cmd = subparsers.add_parser(
        "benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a directory containing benchmark images and does compression"
        "evaluation over benchmark images",
    )
    # Performance tuning parameters
    benchmark_cmd.add_argument(
        "--allow-growth",
        help="Whether to enable allow_growth in GPU_Options",
        default=True,
        type=bool,
    )

    # Arguments for both 'compress' and 'decompress'.
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
        cmd.add_argument("input_file", help="Input filename.")
        cmd.add_argument(
            "output_file",
            nargs="?",
            help="Output filename (optional). If not provided, appends '{}' to "
            "the input filename.".format(ext),
        )

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    if args.command == "train":
        train.train(args)
    elif args.command == "compress":
        if not args.output_file:
            args.output_file = args.input_file + ".ncf"
        test.compress(args)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file = args.input_file + ".png"
        test.decompress(args)
    elif args.command == "benchmark":
        # Set up benchmarks
        benchmarks = [Benchmark(args, "data/benchmark/kodak", name="Kodak")]
        # benchmarks = [
        #     Benchmark(args, "data/benchmark/urban16", name="Urban16"),
        #     Benchmark(args, "data/benchmark/kodak", name="Kodak"),
        #     Benchmark(args, "data/benchmark/set5", name="Set5"),
        #     Benchmark(args, "data/benchmark/set14", name="Set14"),
        #     Benchmark(args, "data/benchmark/bsd100", name="BSD100"),
        # ]
        print(">>>>> Benchmark Validation enabled.....Validating Benchmarks now....>>>>>>")
        for benchmark in benchmarks:
            benchmark.benchmarkcompression()


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
