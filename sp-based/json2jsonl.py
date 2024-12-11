import json
import os
import argparse
from argparse import RawTextHelpFormatter
import sys



def print_error(message):
    print(f"\033[31m{message}\033[0m")  # Print error message in red color

def json_to_jsonl(input_file, output_directory):
    """
    Converts a JSON file to JSONL format and saves it in the specified directory.

    Args:
        input_file (str): Path to the input JSON file.
        output_directory (str): Path to the output directory where JSONL file will be saved.
    """
    # Check if the input file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the input JSON file
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)  # Assuming the JSON file is an array of objects

    # Ensure data is a list (required for JSONL format)
    if not isinstance(data, list):
        raise ValueError(f"Error in converting {input_file}: Input JSON file must contain an array of objects.")

    # Define the output file path
    output_file = os.path.join(output_directory, os.path.splitext(os.path.basename(input_file))[0] + '.jsonl')

    # Write to the JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in data:
            json_line = json.dumps(item)
            outfile.write(json_line + '\n')

    print(f"Successfully converted {input_file} to {output_file}")

def process_default_mode():
    """
    Processes files in the default mode, converting predefined datasets to JSONL.
    """
    datasets = ["train.json", "test.json", "val.json"]
    input_directory = "datasets"

    for dataset in datasets:
        input_file = os.path.join(input_directory, dataset)
        output_directory = input_directory

        try:
            json_to_jsonl(input_file, output_directory)
        except FileNotFoundError as e:
            print_error(e)
        except ValueError as e:
            print_error(e)
        except Exception as e:
            print_error(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=(
            "This script converts JSON files to JSONL format.\n"
            "There are two modes of operation:\n"
            "1. Default mode: Automatically converts predefined json files ('train.json', 'test.json', 'val.json') "
            "in the './datasets/' directory to JSONL format.\n"
            "2. Custom mode: Converts a user-specified input JSON file to JSONL format and saves it to a specified output directory. "
            "In custom mode, both '--input_file' and '--output_directory' are required."
        ),
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("--mode", required=True, choices=["default", "custom"], help="Mode of operation: 'default' or 'custom'.")
    parser.add_argument("--input_file", help="Path to the input JSON file (required in custom mode).")
    parser.add_argument("--output_directory", help="Path to the output directory (required in custom mode).")

    args = parser.parse_args()

    if args.mode == "default":
        process_default_mode()
    elif args.mode == "custom":
        if not args.input_file or not args.output_directory:
            print_error("In custom mode, both --input_file and --output_directory are required.")
            parser.print_help()
        else:
            try:
                json_to_jsonl(args.input_file, args.output_directory)
            except FileNotFoundError as e:
                print_error(e)
            except ValueError as e:
                print_error(e)
            except Exception as e:
                print_error(f"Unexpected error: {e}")