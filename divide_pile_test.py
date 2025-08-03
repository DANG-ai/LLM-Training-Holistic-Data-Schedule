import lm_dataformat as lmd
import tqdm
import json
import os
import time
import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process test Pile dataset and categorize by domain')

    # Define required arguments
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input .jsonl.zst file to process')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for domain-separated files')

    # Parse command line arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    buffer = {}  # Dictionary to hold texts grouped by domain
    print(f"------------------File Test-------------------")

    # Read and process each record in the input file
    for data in tqdm.tqdm(filter(lambda x: x, lmd.Reader(args.input_file).stream_data(get_meta=True))):
        text, meta = data
        domain = meta["pile_set_name"]

        # Initialize domain list if not exists
        if domain not in buffer:
            buffer[domain] = []

        # Add text with metadata to corresponding domain
        buffer[domain].append(json.dumps({"text": text, "meta": meta}))

    # Write grouped data to domain-specific output files
    time1 = time.time()
    for domain, texts in buffer.items():
        output_path = os.path.join(os.path.join(args.output_dir, "train"), f"{domain}.jsonl")

        # Append to domain file
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(texts) + "\n")

        print(f"Write {domain} to {output_path}")

    print(f"--- Writing used {time.time() - time1} seconds ---")


if __name__ == "__main__":
    main()