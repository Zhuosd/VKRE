import argparse
import bz2
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def open_tsv_bz2(path):
    return bz2.open(path, "rt", encoding="utf-8", errors="replace", newline="")


def load_audio_features(path):
    features = {}
    with open_tsv_bz2(path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        dim = len(header) - 1
        for row in reader:
            if not row:
                continue
            features[row[0]] = np.asarray([float(x) for x in row[1:]], dtype=np.float32)
    return features, dim


def count_entities(interactions_path, valid_tracks):
    user_counts = Counter()
    track_counts = Counter()
    with open_tsv_bz2(interactions_path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            user_id, track_id, count = row[0], row[1], int(row[2])
            if track_id not in valid_tracks:
                continue
            user_counts[user_id] += count
            track_counts[track_id] += count
    return user_counts, track_counts


def select_users(interactions_path, selected_tracks, max_users, min_interactions):
    user_item_counts = Counter()
    with open_tsv_bz2(interactions_path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            user_id, track_id = row[0], row[1]
            if track_id in selected_tracks:
                user_item_counts[user_id] += 1

    eligible = [(u, c) for u, c in user_item_counts.items() if c >= min_interactions]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    return {u for u, _ in eligible[:max_users]}


def collect_interactions(interactions_path, selected_users, selected_tracks):
    interactions = defaultdict(list)
    with open_tsv_bz2(interactions_path) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            user_id, track_id, count = row[0], row[1], int(row[2])
            if user_id in selected_users and track_id in selected_tracks:
                interactions[user_id].append((track_id, count))
    return {u: items for u, items in interactions.items() if len(items) >= 3}


def split_user_items(items):
    items = sorted(items, key=lambda x: (-x[1], x[0]))
    n = len(items)
    train_end = max(1, int(n * 0.8))
    val_end = max(train_end + 1, int(n * 0.9))
    if val_end >= n:
        val_end = n - 1
    return items[:train_end], items[train_end:val_end], items[val_end:]


def load_item_features(genres_path, selected_track_to_idx, n_items, max_genres_per_item):
    item_features = {idx: [idx] for idx in selected_track_to_idx.values()}
    genre_to_idx = {}
    with open_tsv_bz2(genres_path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        genres = header[1:]
        for row in reader:
            if not row:
                continue
            track_id = row[0]
            if track_id not in selected_track_to_idx:
                continue
            scored = []
            for col_idx, value in enumerate(row[1:]):
                try:
                    score = float(value)
                except ValueError:
                    score = 0.0
                if score > 0:
                    scored.append((score, genres[col_idx]))
            scored.sort(reverse=True)
            feature_ids = [selected_track_to_idx[track_id]]
            for _, genre in scored[:max_genres_per_item]:
                if genre not in genre_to_idx:
                    genre_to_idx[genre] = len(genre_to_idx)
                feature_ids.append(n_items + genre_to_idx[genre])
            item_features[selected_track_to_idx[track_id]] = feature_ids
    return {str(k): v for k, v in item_features.items()}, genre_to_idx


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("raw/music4all_onion"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/music4all"))
    parser.add_argument("--src_dir", type=Path, default=Path("src"))
    parser.add_argument("--max_users", type=int, default=800)
    parser.add_argument("--max_items", type=int, default=2000)
    parser.add_argument("--min_interactions", type=int, default=10)
    parser.add_argument("--max_genres_per_item", type=int, default=5)
    args = parser.parse_args()

    audio_path = args.raw_dir / "id_musicnn.tsv.bz2"
    genres_path = args.raw_dir / "id_genres_tf-idf.tsv.bz2"
    interactions_path = args.raw_dir / "userid_trackid_count.tsv.bz2"

    print("Loading real audio-derived MusicNN features...")
    audio_features, audio_dim = load_audio_features(audio_path)
    print(f"Audio features: {len(audio_features)} tracks, dim={audio_dim}")

    print("Counting listening interactions...")
    _, track_counts = count_entities(interactions_path, set(audio_features))
    selected_tracks = {track for track, _ in track_counts.most_common(args.max_items)}

    print("Selecting users...")
    selected_users = select_users(
        interactions_path,
        selected_tracks,
        max_users=args.max_users,
        min_interactions=args.min_interactions,
    )

    print("Collecting user-track interactions...")
    interactions = collect_interactions(interactions_path, selected_users, selected_tracks)

    selected_users = sorted(interactions)
    selected_tracks = sorted({track for items in interactions.values() for track, _ in items})
    user_to_idx = {u: idx for idx, u in enumerate(selected_users)}
    track_to_idx = {t: idx for idx, t in enumerate(selected_tracks)}

    train_json, val_json, test_json = {}, {}, {}
    user_his_train, user_his_val, user_his_test = {}, {}, {}
    user_audio = {}
    user_labels = {}

    for user_id in selected_users:
        uid = user_to_idx[user_id]
        train_items, val_items, test_items = split_user_items(interactions[user_id])
        splits = [
            (train_items, train_json, user_his_train),
            (val_items, val_json, user_his_val),
            (test_items, test_json, user_his_test),
        ]
        weighted = []
        weights = []
        for split_items, split_json, split_his in splits:
            entries = []
            ids = []
            for track_id, count in split_items:
                iid = track_to_idx[track_id]
                entries.append([str(iid), f"{track_id}.wav"])
                ids.append(iid)
                if split_json is train_json:
                    weighted.append(audio_features[track_id] * count)
                    weights.append(count)
            split_json[str(uid)] = entries
            split_his[str(uid)] = ids
        if weighted:
            user_audio[str(uid)] = (np.sum(weighted, axis=0) / max(1, sum(weights))).astype(float).tolist()
        else:
            user_audio[str(uid)] = np.zeros(audio_dim, dtype=float).tolist()
        user_labels[str(uid)] = [[0, 0] for _ in train_json[str(uid)]]

    print("Building item genre features...")
    item_features, genre_to_idx = load_item_features(
        genres_path,
        track_to_idx,
        len(selected_tracks),
        args.max_genres_per_item,
    )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "info.txt").write_text(f"{len(selected_users)} {len(selected_tracks)}\n")
    (args.data_dir / "user_list.txt").write_text("\n".join(selected_users) + "\n")
    (args.data_dir / "item_list.txt").write_text("\n".join(selected_tracks) + "\n")
    write_json(args.data_dir / "train.json", train_json)
    write_json(args.data_dir / "val.json", val_json)
    write_json(args.data_dir / "test.json", test_json)
    write_json(args.data_dir / "user_his_train.json", user_his_train)
    write_json(args.data_dir / "user_his_val.json", user_his_val)
    write_json(args.data_dir / "user_his_test.json", user_his_test)
    write_json(args.data_dir / "item_features.json", item_features)
    write_json(args.data_dir / "genre_to_idx.json", genre_to_idx)

    write_json(args.src_dir / "emb" / "emb_sum_music4all.json", user_audio)
    write_json(args.src_dir / "user_pred_labels_music4all.json", user_labels)

    print("Done.")
    print(f"Users: {len(selected_users)}")
    print(f"Items: {len(selected_tracks)}")
    print(f"Genres used: {len(genre_to_idx)}")
    print(f"Train interactions: {sum(len(v) for v in train_json.values())}")
    print(f"Val interactions: {sum(len(v) for v in val_json.values())}")
    print(f"Test interactions: {sum(len(v) for v in test_json.values())}")


if __name__ == "__main__":
    main()
