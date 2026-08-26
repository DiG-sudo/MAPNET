import argparse
import re
from pathlib import Path

import numpy as np

from data import load_config, project_path


NCP = {
    "A": [1, 1, 1],
    "U": [0, 0, 1],
    "C": [0, 1, 0],
    "G": [1, 0, 0],
}

DPCP = {
    "AA": [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.04],
    "AU": [-0.06, -1.36, 3.24, 1.1, 7.1, 33, -15.4, -5.7, -15.5, -1.1, 0.14],
    "AC": [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.14],
    "AG": [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.08],
    "UA": [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -13.3, -35.5, -2.35, 0.1],
    "UU": [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.27],
    "UC": [0.07, -1.39, 3.22, 0, 6.1, 35, -16.9, -14.2, -34.9, -3.42, 0.26],
    "UG": [-0.01, -1.78, 3.32, 0.3, 12.1, 32, -11.1, -12.2, -29.7, -3.26, 0.17],
    "CA": [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -10.5, -27.8, -2.11, 0.21],
    "CU": [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.52],
    "CC": [-0.01, -1.78, 3.32, 0.3, 8.7, 32, -11.1, -12.2, -29.7, -3.26, 0.49],
    "CG": [0.3, -1.89, 3.3, -0.1, 12.1, 27, -15.6, -8, -19.4, -2.36, 0.35],
    "GA": [-0.02, -1.45, 3.26, -0.2, 10.7, 32, -16, -8.1, -22.6, -1.33, 0.21],
    "GU": [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.44],
    "GC": [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -10.2, -26.2, -2.35, 0.48],
    "GG": [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -7.6, -19.2, -2.11, 0.34],
}


def load_word2vec_text(path, expected_dim):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CircRNA2Vec weights not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        vocab_size, vector_dim = map(int, handle.readline().split())
        if vector_dim != expected_dim:
            raise ValueError(f"Expected {expected_dim}-D weights, found {vector_dim}-D")
        vectors = np.empty((vocab_size, vector_dim), dtype=np.float32)
        token_to_index = {}
        for index, line in enumerate(handle):
            fields = line.rstrip().split()
            if len(fields) != vector_dim + 1:
                raise ValueError(f"Malformed weight row {index + 2} in {path}")
            token_to_index[fields[0]] = index
            vectors[index] = np.asarray(fields[1:], dtype=np.float32)
    if len(token_to_index) != vocab_size:
        raise ValueError(f"Weight header declares {vocab_size} tokens, read {len(token_to_index)}")
    return token_to_index, vectors


def read_fasta(path):
    records = []
    header = None
    sequence_parts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_parts)))
                header = line[1:]
                sequence_parts = []
            elif header is None:
                raise ValueError(f"Sequence found before header in {path}")
            else:
                sequence_parts.append(line)
    if header is not None:
        records.append((header, "".join(sequence_parts)))
    return records


def normalize_centered(sequence, length):
    sequence = sequence.upper().replace("T", "U")
    half = length // 2
    start = len(sequence) // 2 - half
    chars = []
    for offset in range(length):
        position = start + offset
        chars.append(sequence[position] if 0 <= position < len(sequence) else "N")
    return "".join(chars)


def binding_index(header, center=50, max_index=100):
    match = re.search(r"start:(\d+),end:(\d+)", header)
    if match is None:
        start, end = 0, 0
    else:
        start, end = map(int, match.groups())
    length = end - start
    left = max(0, center - length // 2)
    right = min(max_index, center + (length - length // 2))
    return ",".join(str(index) for index in range(left, right + 1))


def circRNA2vec_features(sequences, token_to_index, vectors, kmer_size, length):
    output = np.empty((len(sequences), length, vectors.shape[1]), dtype=np.float32)
    padding_vector = vectors[0]
    for row, sequence in enumerate(sequences):
        matched = []
        for start in range(0, len(sequence) - kmer_size + 1):
            index = token_to_index.get(sequence[start : start + kmer_size])
            if index is not None:
                matched.append(index)
        used = min(len(matched), length)
        if used:
            output[row, :used] = vectors[np.asarray(matched[:used], dtype=np.int64)]
        if used < length:
            output[row, used:] = padding_vector
    return output


def sequence_features(sequences, length):
    lookup = np.zeros(256, dtype=np.int64)
    for index, base in enumerate("ACGU"):
        lookup[ord(base)] = index
        lookup[ord(base.lower())] = index
    encoded = lookup[np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)].reshape(-1, length)
    eye4 = np.eye(4, dtype=np.float32)
    onehot = eye4[encoded]
    kmer1_frequency = onehot.sum(axis=1, dtype=np.float32) / float(length)
    kmer1 = onehot * kmer1_frequency[:, None, :]

    dimer_ids = encoded[:, :-1] * 4 + encoded[:, 1:]
    dimer_onehot = np.eye(16, dtype=np.float32)[dimer_ids]
    kmer2 = np.zeros((len(sequences), length, 16), dtype=np.float32)
    kmer2_frequency = dimer_onehot.sum(axis=1, dtype=np.float32) / float(length - 1)
    kmer2[:, :-1] = dimer_onehot * kmer2_frequency[:, None, :]

    trimer_ids = encoded[:, :-2] * 16 + encoded[:, 1:-1] * 4 + encoded[:, 2:]
    trimer_onehot = np.eye(64, dtype=np.float32)[trimer_ids]
    kmer3 = np.zeros((len(sequences), length, 64), dtype=np.float32)
    kmer3_frequency = trimer_onehot.sum(axis=1, dtype=np.float32) / float(length - 2)
    kmer3[:, :-2] = trimer_onehot * kmer3_frequency[:, None, :]

    ncp_table = np.asarray([NCP[base] for base in "ACGU"], dtype=np.float32)
    ncp = ncp_table[encoded]
    dpcp_table = np.empty((16, 11), dtype=np.float32)
    for left_index, left in enumerate("ACGU"):
        for right_index, right in enumerate("ACGU"):
            dpcp_table[left_index * 4 + right_index] = DPCP[left + right]
    dpcp = np.zeros((len(sequences), length, 11), dtype=np.float32)
    dpcp[:, :-1] = dpcp_table[dimer_ids]
    return {"kmer1": kmer1, "kmer2": kmer2, "kmer3": kmer3, "ncp": ncp, "dpcp": dpcp}


def prepare_protein(protein, raw_dir, output_dir, config, token_to_index, vectors):
    records = []
    for label_name, label in (("positive", 1), ("negative", 0)):
        for header, sequence in read_fasta(raw_dir / protein / label_name):
            records.append((header, normalize_centered(sequence, config["sequence_length"]), label))
    sequences = [record[1] for record in records]
    labels = np.asarray([record[2] for record in records], dtype=np.int64)
    sample_ids = np.asarray([binding_index(record[0]) for record in records])
    features = sequence_features(sequences, config["sequence_length"])
    features["circ2vec_embed"] = circRNA2vec_features(
        sequences,
        token_to_index,
        vectors,
        config["kmer_size"],
        config["sequence_length"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{protein}_features.npz"
    np.savez_compressed(
        destination,
        sample_id=sample_ids,
        sequence=np.asarray(sequences),
        label=labels,
        **features,
    )
    print(f"{protein}: {len(labels)} samples -> {destination}")


def main():
    parser = argparse.ArgumentParser(description="Generate MAPNet inputs from the raw circRNA-RBP data.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--proteins", nargs="+", help="Optional subset for a partial run.")
    args = parser.parse_args()
    config = load_config(args.config)
    raw_dir = project_path(config, config["paths"]["raw_data_dir"])
    output_dir = project_path(config, config["paths"]["feature_dir"])
    weights_path = project_path(config, config["paths"]["circRNA2vec_weights"])
    proteins = args.proteins or config["proteins"]
    token_to_index, vectors = load_word2vec_text(
        weights_path, int(config["data"]["circRNA2vec_dim"])
    )
    for protein in proteins:
        prepare_protein(protein, raw_dir, output_dir, config["data"], token_to_index, vectors)


if __name__ == "__main__":
    main()
