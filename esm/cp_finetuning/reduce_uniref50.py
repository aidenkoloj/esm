
import argparse

def filter_fasta(input_path, output_path, max_len):
    kept = 0
    skipped = 0
    header = None
    seq = []

    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if line.startswith(">"):
                # Write previous sequence if it passes filter
                if header is not None:
                    full_seq = "".join(seq)
                    if len(full_seq) <= max_len:
                        f_out.write(header + "\n")
                        f_out.write(full_seq + "\n")
                        kept += 1
                    else:
                        skipped += 1
                header = line
                seq = []
            else:
                seq.append(line)

        # Don't forget the last sequence
        if header is not None:
            full_seq = "".join(seq)
            if len(full_seq) <= max_len:
                f_out.write(header + "\n")
                f_out.write(full_seq + "\n")
                kept += 1
            else:
                skipped += 1

    print(f"Kept: {kept:,} sequences")
    print(f"Skipped: {skipped:,} sequences (>{max_len} residues)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="uniref50.fasta")
    parser.add_argument("--output", type=str, default="uniref50_filtered512_length.fasta")
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()
    filter_fasta(args.input, args.output, args.max_len)