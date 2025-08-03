import lm_dataformat as lmd
import tqdm
import json
import os
import time
import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process Pile dataset and categorize by domain')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing Pile dataset files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for domain-separated files')

    # Parse command line arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)

    # Generate list of input filenames (00.jsonl.zst to 29.jsonl.zst)
    fnames = [os.path.join(args.input_dir, f"{str(i).zfill(2)}.jsonl.zst") for i in range(30)]

    # Process each input file
    for f_im, fname in enumerate(fnames):
        buffer = {}  # Dictionary to hold texts grouped by domain
        print(f"------------------File {str(f_im).zfill(2)}-------------------")

        # Read and process each record in the current file
        for data in tqdm.tqdm(filter(lambda x: x, lmd.Reader(fname).stream_data(get_meta=True))):
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
            output_path = os.path.join(os.path.join(args.output_dir, "test"), f"{domain}.jsonl")

            # Append to domain file
            with open(output_path, "a", encoding="utf-8") as f:
                f.write("\n".join(texts) + "\n")

            print(f"Write {domain} to {output_path}")

        print(f"--- Writing used {time.time() - time1} seconds ---")


if __name__ == "__main__":
    main()