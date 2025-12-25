#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import tarfile
import tempfile
import urllib.request

DEFAULT_URL = (
    "https://downloads.wortschatz-leipzig.de/corpora/"
    "cat_wikipedia_2021_300K.tar.gz"
)

LETTERS = set("abcdefghijklmnopqrstuvwxyzàáèéíïòóúüçñ")
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


def get_features(word: str, digraphs) -> set:
    features = set()
    for ch in word:
        if ch in LETTERS:
            features.add(f"letter:{ch}")
        elif ch in SYMBOLS:
            features.add(f"symbol:{ch}")
    for d in digraphs:
        if d in word:
            features.add(f"digraph:{d}")
    return features


def select_words(
    words,
    target_size: int,
    seed_size: int,
    candidate_size: int,
    allowed_letters,
    allowed_symbols,
    required_digraphs,
    include_words,
):
    candidates = words[:candidate_size] if candidate_size else words
    selected = []
    selected_set = set()

    for word in include_words:
        if word in selected_set:
            continue
        selected.append(word)
        selected_set.add(word)
        if len(selected) >= target_size:
            return selected, set()

    for word, _freq in candidates:
        if word in selected_set:
            continue
        selected.append(word)
        selected_set.add(word)
        if len(selected) >= seed_size:
            break

    target_features = {f"letter:{ch}" for ch in allowed_letters}
    target_features |= {f"symbol:{ch}" for ch in allowed_symbols}
    target_features |= {f"digraph:{d}" for d in required_digraphs}

    covered = set()
    for word in selected:
        covered |= get_features(word, required_digraphs)

    missing = target_features - covered

    while missing and len(selected) < target_size:
        best_word = None
        best_gain = 0
        for word, _freq in candidates:
            if word in selected_set:
                continue
            gain = len(get_features(word, required_digraphs) & missing)
            if gain > best_gain:
                best_gain = gain
                best_word = word
        if best_word is None:
            break
        selected.append(best_word)
        selected_set.add(best_word)
        missing -= get_features(best_word, required_digraphs)

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


def load_levels_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_level_words(words, levels_config, candidate_size: int, output_dir: str):
    levels = levels_config.get("levels", [])
    if not levels:
        raise RuntimeError("No levels defined in levels config.")

    allowed_letters = set()
    allowed_symbols = set()
    required_digraphs = []
    outputs = []

    for level in levels:
        name = level["name"]
        target_size = int(level.get("target_size", 800))
        seed_size = int(level.get("seed_size", 200))
        add_chars = level.get("add_chars", "")
        add_symbols = level.get("add_symbols", "")
        add_digraphs = level.get("add_digraphs", [])
        include_words = level.get("include_words", [])

        allowed_letters |= set(add_chars)
        allowed_symbols |= set(add_symbols)
        required_digraphs.extend(add_digraphs)

        filtered = []
        for word, freq in words:
            if all((ch in allowed_letters) or (ch in allowed_symbols) for ch in word):
                filtered.append((word, freq))

        normalized_include = []
        for word in include_words:
            w = word.lower()
            if all((ch in allowed_letters) or (ch in allowed_symbols) for ch in w):
                normalized_include.append(w)

        if not filtered:
            outputs.append((name, [], set()))
            continue

        if len(filtered) < target_size:
            target_size = len(filtered)

        selected, missing = select_words(
            filtered,
            target_size,
            min(seed_size, target_size),
            candidate_size,
            allowed_letters,
            allowed_symbols,
            required_digraphs,
            normalized_include,
        )

        output_path = os.path.join(output_dir, f"{name}.txt")
        outputs.append((output_path, selected, missing))

    return outputs


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
    parser.add_argument("--levels-config", default="")
    parser.add_argument("--levels-output-dir", default="dictionary")
    parser.add_argument(
        "--levels-source-output", default="dictionary/ca.levels.source.txt"
    )
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

    if args.levels_config:
        levels_config = load_levels_config(args.levels_config)
        level_outputs = build_level_words(
            words, levels_config, args.candidate_size, args.levels_output_dir
        )
        for output_path, selected, missing in level_outputs:
            if not selected:
                print(f"Skipped {output_path} (no candidates)")
                continue
            write_output(output_path, selected)
            if missing:
                missing_list = sorted(missing)
                print(f"Missing coverage for {output_path}: {', '.join(missing_list)}")
            print(f"Wrote {len(selected)} words to {output_path}")
        write_source_info(args.levels_source_output, args.url)
    else:
        selected, missing = select_words(
            words,
            args.target_size,
            args.seed_size,
            args.candidate_size,
            LETTERS,
            SYMBOLS,
            DIGRAPHS,
            [],
        )

        write_output(args.output, selected)
        write_source_info(args.source_output, args.url)

        if missing:
            missing_list = sorted(missing)
            print("Missing coverage:", ", ".join(missing_list))
        print(f"Wrote {len(selected)} words to {args.output}")


if __name__ == "__main__":
    main()
