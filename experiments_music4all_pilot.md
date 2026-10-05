# Music4All-Onion Real-Data Pilot

Date: 2026-07-27

## Goal

Validate whether VKRE-style user-side audio representations provide useful recommendation signal on a public real dataset with real listening interactions and real audio-derived features.

## Dataset

Source: Music4All-Onion, a public extension of Music4All.

Files used:

- `raw/music4all_onion/id_musicnn.tsv.bz2`: real audio-derived MusicNN features, 50 dimensions.
- `raw/music4all_onion/id_genres_tf-idf.tsv.bz2`: genre TF-IDF item features.
- `raw/music4all_onion/userid_trackid_count.tsv.bz2`: real Last.fm user-track listening counts.

Pilot subset:

- Users: 800
- Items: 2000
- Genres used: 168
- Train interactions: 495230
- Validation interactions: 61889
- Test interactions: 62309

Construction:

- User audio representation is the weighted mean of MusicNN features from each user's training tracks.
- Interactions are split per user into train/validation/test.
- Item-side feature indices are built from genre TF-IDF features.
- The current model requires user profile labels, so dummy labels are used for this pilot. These labels are not interpreted as semantic user attributes.

## Commands

Run from `VKER-Code/src`.

```bash
python main.py --dataset music4all --epochs 5 --batch_size 4096 --recdim 50 --device cpu --test 1 --topk 10 --patience 5 --ens_ratio 0.5 --lr 0.001 --weight_decay 0.0001
python main.py --dataset music4all --epochs 5 --batch_size 4096 --recdim 50 --device cpu --test 1 --topk 10 --patience 5 --ens_ratio 1.0 --lr 0.001 --weight_decay 0.0001
python main.py --dataset music4all --epochs 5 --batch_size 4096 --recdim 50 --device cpu --test 1 --topk 10 --patience 5 --ens_ratio 0.0 --lr 0.001 --weight_decay 0.0001
```

## Results

| Setting | `ens_ratio` | Best validation signal | Test F1 | Test P@10 | Test R@10 | Test NDCG@10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Fused ID/profile + audio user representation | 0.5 | Epoch 1: P=0.152375, R=0.019564, NDCG=0.154120 | 0.020596 | 0.091000 | 0.011612 | 0.091164 |
| No-audio control | 1.0 | Epoch 1: P=0.147625, R=0.018942, NDCG=0.146789 | 0.021041 | 0.092625 | 0.011868 | 0.100175 |
| Audio-only user representation | 0.0 | Epoch 5: P=0.160875, R=0.020411, NDCG=0.165355 | 0.018866 | 0.083250 | 0.010639 | 0.083385 |

## Interpretation

This pilot partially validates the presence of useful audio signal: the audio-only setting achieves the strongest validation metrics among the three settings. However, the current test results do not support the stronger claim that adding audio improves final recommendation accuracy, because the no-audio control is best on test P@10, R@10, F1, and NDCG@10.

The most likely issue is not that audio features are useless, but that the current fusion path and hyperparameters are not tuned for this real-data construction. The fixed `ens_ratio=0.5` fusion underperforms the no-audio control on test, while audio-only is promising on validation but weaker on test. The next main-experiment step should therefore tune fusion weights, normalize user/audio embeddings more carefully, repeat over multiple seeds, and compare against stronger item-audio/content baselines.

