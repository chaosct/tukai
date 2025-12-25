#!/usr/bin/env python3
import argparse
import datetime as dt
import io
import os
import tarfile
import tempfile
import urllib.request

DEFAULT_URL = (
    "https://downloads.wortschatz-leipzig.de/corpora/"
    "cat_wikipedia_2021_300K.tar.gz"
)

LETTERS = set("abcdefghijklmnopqrstuvwxyzàáèéíïòóúüç")
SYMBOLS = set(["'", "’", "-", "·"])
DIGRAPHS = [
    "ny",
    "ll",
    "rr",
    "qu",
    "qü",
    "gu",
    "gü",
    "ig",
    "ix",
    "tx",
    "tg",
    "tj",
    "ss",
    "l·l",
]


def download_to_temp(url: str) -> str:
    fd, path = tempfile.mkstemp(prefix="tukai_ca_", suffix=".tar.gz")
    os.close(fd)
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return path


def find_words_member(tf: tarfile.TarFile) -> tarfile.TarInfo:
    candidates = [m for m in tf.getmembers() if m.name.endswith("-words.txt")]
    if not candidates:
        raise RuntimeError("No -words.txt file found in the corpus archive.")
    # Prefer the shortest path in case of multiple candidates.
    return sorted(candidates, key=lambda m: len(m.name))[0]


def parse_words(tf: tarfile.TarFile, member: tarfile.TarInfo):
    with tf.extractfile(member) as f:
        for raw in f:
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            word = parts[1].lower()
            try:
                freq = int(parts[2])
            except ValueError:
                continue
            if not is_valid_word(word):
                continue
            yield word, freq


def is_valid_word(word: str) -> bool:
    if not word:
        return False
    has_letter = False
    for ch in word:
        if ch in LETTERS or ch in SYMBOLS:
            if ch in LETTERS:
                has_letter = True
            continue
        return False
    return has_letter


def get_features(word: str) -> set:
    features = set()
    for ch in word:
        if ch in LETTERS:
            features.add(f"letter:{ch}")
        elif ch in SYMBOLS:
            features.add(f"symbol:{ch}")
    for d in DIGRAPHS:
        if d in word:
            features.add(f"digraph:{d}")
    return features


def select_words(words, target_size: int, seed_size: int, candidate_size: int):
    candidates = words[:candidate_size]
    selected = []
    selected_set = set()

    for word, _freq in candidates:
        if word in selected_set:
            continue
        selected.append(word)
        selected_set.add(word)
        if len(selected) >= seed_size:
            break

    target_features = {f"letter:{ch}" for ch in LETTERS}
    target_features |= {f"symbol:{ch}" for ch in SYMBOLS}
    target_features |= {f"digraph:{d}" for d in DIGRAPHS}

    covered = set()
    for word in selected:
        covered |= get_features(word)

    missing = target_features - covered

    while missing and len(selected) < target_size:
        best_word = None
        best_gain = 0
        for word, _freq in candidates:
            if word in selected_set:
                continue
            gain = len(get_features(word) & missing)
            if gain > best_gain:
                best_gain = gain
                best_word = word
        if best_word is None:
            break
        selected.append(best_word)
        selected_set.add(best_word)
        missing -= get_features(best_word)

    for word, _freq in candidates:
        if len(selected) >= target_size:
            break
        if word in selected_set:
            continue
        selected.append(word)
        selected_set.add(word)

    return selected, missing


def write_output(path: str, words):
    with open(path, "w", encoding="utf-8") as f:
        for w in words:
            f.write(f"{w}\n")


def write_source_info(path: str, url: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Source: Leipzig Corpora Collection\n")
        f.write(f"URL: {url}\n")
        f.write(f"Generated: {dt.datetime.utcnow().isoformat()}Z\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build Catalan dictionary from a frequency corpus."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Corpus URL (tar.gz).")
    parser.add_argument("--target-size", type=int, default=1200)
    parser.add_argument("--seed-size", type=int, default=400)
    parser.add_argument("--candidate-size", type=int, default=20000)
    parser.add_argument("--output", default="dictionary/ca.txt")
    parser.add_argument("--source-output", default="dictionary/ca.source.txt")
    args = parser.parse_args()

    if args.seed_size > args.target_size:
        raise SystemExit("seed-size cannot be larger than target-size.")

    archive_path = download_to_temp(args.url)
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            member = find_words_member(tf)
            words = list(parse_words(tf, member))
    finally:
        os.remove(archive_path)

    selected, missing = select_words(
        words, args.target_size, args.seed_size, args.candidate_size
    )

    write_output(args.output, selected)
    write_source_info(args.source_output, args.url)

    if missing:
        missing_list = sorted(missing)
        print("Missing coverage:", ", ".join(missing_list))
    print(f"Wrote {len(selected)} words to {args.output}")


if __name__ == "__main__":
    main()
