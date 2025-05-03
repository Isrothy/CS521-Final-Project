#!/usr/bin/env python2
from __future__ import print_function

import argparse
import binascii
import common_libs.utilities as ut
import copy
import data.data_cost as dt
import ithemal_utils
import multiprocessing
import os
import subprocess
import sys
import threading
import torch
import torch.nn.functional as F
import warnings
import csv
import time
import Queue

_TOKENIZER = os.path.join(
    os.environ.get("ITHEMAL_HOME", "."), "data_collection", "build", "bin", "tokenizer"
)


def load_model_and_data(model_file, model_data_file):
    """Loads the model architecture and trained weights."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)
        (model, data) = ithemal_utils.load_model_and_data(model_file)

    state_dict = torch.load(model_data_file)
    model_dict = model.state_dict()

    new_model_dict = {
        k: v
        for (k, v) in state_dict.get("model", state_dict).items()
        if k in model_dict
    }
    model_dict.update(new_model_dict)
    model.load_state_dict(model_dict)

    model.eval()

    return (model, data)


def datum_of_code(data, block_hex):
    xml = subprocess.check_output([_TOKENIZER, block_hex, "--token"])
    intel = ""
    data.raw_data = [(-1, -1, intel, xml)]
    data.data = []
    data.prepare_data(fixed=True, progress=False)
    return data.data[-1]

def process_line_worker(model_file, model_data_file, input_queue, output_queue):
    try:
        model, data = load_model_and_data(model_file, model_data_file)

        while True:
            line_tuple = input_queue.get()
            if line_tuple is None:
                break

            line_num, line = line_tuple
            if not line:
                continue

            parts = line.strip().split(",")
            if len(parts) != 2:
                output_queue.put((None, None))
                continue

            block_hex, true_val_str = parts
            try:
                true_value = float(true_val_str)
            except ValueError:
                output_queue.put((None, None))
                continue

            datum = datum_of_code(data, block_hex)

            if datum is None:
                output_queue.put((None, None))
                continue

            try:
                with torch.no_grad():
                    prediction_tensor = model(datum)

                prediction = prediction_tensor.item()
                output_queue.put((prediction, true_value))
                if hasattr(model, "remove_refs"):
                    model.remove_refs(datum)

            except Exception as e:
                output_queue.put((None, None))

    except Exception as e:
        print("FATAL: Worker process failed: {}".format(e), file=sys.stderr)
        output_queue.put(None)


def collect_results(output_queue, total_lines):
    """Collects results from the output queue."""
    results = []
    processed_count = 0
    while processed_count < total_lines:
        result = output_queue.get()
        if result is None:
            print(
                "Warning: Received early termination signal from a worker.",
                file=sys.stderr,
            )
            pass
        else:
            results.append(result)
        processed_count += 1
        print("\rProcessed: {}/{}".format(processed_count, total_lines), end="")
        sys.stdout.flush()
    print()
    return results


def calculate_loss(results):
    predictions = []
    true_values = []
    valid_count = 0
    fail_count = 0
    zero_true_count = 0

    for pred, true in results:
        if pred is not None and true is not None:
            if abs(true) < 1e-9:
                zero_true_count += 1
                valid_count += 1
            else:
                predictions.append(pred)
                true_values.append(true)
                valid_count += 1
        else:
            fail_count += 1

    if not predictions:
        print(
            "Error: No valid predictions with non-zero true values were made.",
            file=sys.stderr,
        )
        return None, valid_count, fail_count, zero_true_count

    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    true_values_tensor = torch.tensor(true_values, dtype=torch.float32)

    absolute_percentage_error = torch.abs(
        (predictions_tensor - true_values_tensor) / true_values_tensor
    )

    mape_loss = torch.mean(absolute_percentage_error) * 100

    return mape_loss.item(), valid_count, fail_count, zero_true_count


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Ithemal model performance against a CSV dataset."
    )
    parser.add_argument(
        "--model", help="Path to the model architecture file (.dump)", required=True
    )
    parser.add_argument(
        "--model-data",
        help="Path to the trained model weights file (.mdl)",
        required=True,
    )
    parser.add_argument(
        "--input-file",
        help="Path to the input CSV file (format: hex_code,true_value)",
        required=True,
    )
    parser.add_argument(
        "--parallel",
        help="Number of parallel worker processes",
        type=int,
        default=multiprocessing.cpu_count(),
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(
            "Error: Input file not found: {}".format(args.input_file), file=sys.stderr
        )
        sys.exit(1)
    if not os.path.exists(args.model):
        print("Error: Model file not found: {}".format(args.model), file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.model_data):
        print(
            "Error: Model data file not found: {}".format(args.model_data),
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.exists(_TOKENIZER):
        print(
            "Error: Tokenizer executable not found at {}".format(_TOKENIZER),
            file=sys.stderr,
        )
        print(
            "Ensure ITHEMAL_HOME is set correctly and the tokenizer is built.",
            file=sys.stderr,
        )
        sys.exit(1)

    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    print("Starting {} worker processes...".format(args.parallel))
    workers = []
    for _ in range(args.parallel):
        p = multiprocessing.Process(
            target=process_line_worker,
            args=(args.model, args.model_data, input_queue, output_queue),
        )
        p.daemon = True
        p.start()
        workers.append(p)

    print("Reading input file: {}".format(args.input_file))
    line_count = 0
    total_lines = 0
    try:
        with open(args.input_file, "r") as infile:
            initial_pos = infile.tell()
            total_lines = sum(1 for _ in infile) + line_count
            infile.seek(initial_pos)
            print("Found {} lines to process.".format(total_lines))

            for i, line in enumerate(infile):
                input_queue.put((i + 1, line))
                line_count += 1

    except IOError as e:
        print("Error reading input file: {}".format(e), file=sys.stderr)
        for _ in range(args.parallel):
            try:
                input_queue.put(None, block=False)
            except Queue.Full:
                pass
        sys.exit(1)
    except Exception as e:
        print("Unexpected error reading input file: {}".format(e), file=sys.stderr)
        sys.exit(1)

    print("Finished reading input. Signaling workers to stop...")
    for _ in range(args.parallel):
        input_queue.put(None)

    print("Collecting results...")
    results = collect_results(output_queue, max(0, total_lines))

    print("Waiting for workers to terminate...")
    for p in workers:
        p.join(timeout=60)

    print("Calculating final loss...")
    final_loss, valid_count, fail_count, zero_true_count = calculate_loss(results)

    print("\n--- Results ---")
    print("Total lines processed: {}".format(max(0, total_lines)))
    print("Successfully predicted (incl. zero true values): {}".format(valid_count))
    print("Failed/Skipped lines: {}".format(fail_count))
    print("Lines skipped due to zero true value: {}".format(zero_true_count))
    if final_loss is not None:
        print("Mean Absolute Percentage Error (MAPE): %.4f%%" % final_loss)
    else:
        print("Could not calculate loss (no valid non-zero true values).")


if __name__ == "__main__":
    main()
