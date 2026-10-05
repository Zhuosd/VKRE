# Voice-Aware Recommendation with Knowledge Graph Enhancement

## Abstract
Recommender systems traditionally rely on text-based inputs such as reviews, item descriptions, or dialogue utterances. While effective, these signals capture only explicit intent and fail to reflect the richer cues present in natural human interactions. With the proliferation of intelligent recommendation assistants, voice has emerged as an important medium, offering expressive signals such as prosody, emotion, and speaking style that enable more adaptive personalization. However, the variability and semantic ambiguity of voice data make it challenging to accurately model user preferences. Moreover, voice alone often lacks sufficient contextual grounding, leading to suboptimal recommendations. To address these issues, we propose VKRE, a unified framework that integrates voice interactions with knowledge graphs. VKRE constructs cross-modal semantic pathways to align heterogeneous representations across voice, text, and knowledge graphs. It further employs graph-based aggregation strategies to refine multimodal embeddings and capture high-order user–voice–item relationships. Extensive experiments on public dialogue datasets used as proxies for voice interactions demonstrate that VKRE consistently outperforms strong baselines, highlighting the potential of incorporating voice semantics into recommendations.

## Reproducibility package
This repository is the reproducibility package for the VKRE paper. It includes the full source code, preprocessing utilities, dataset loaders, model implementation, and experiment commands needed to reproduce the results reported in the manuscript.

The package is organized as follows:
- `src/`: model implementation, data loaders, training pipeline, and evaluation scripts
- `data/`: processed recommendation datasets and user/item histories
- `raw/`: raw or source-form data used by the pipeline
- `scripts/`: auxiliary utilities for dataset preparation and preprocessing
- `experiments_music4all_pilot.md`: pilot-level Music4All experiment notes and commands

## Environment setup
Python 3.10+ is recommended. Install the required dependencies from the project source directory:

```bash
cd VKER-Code/src
pip install -r requirements.txt
```

## Quick start
From the `VKER-Code/src` directory, you can run the main training and evaluation pipeline for the supported datasets:

```bash
python main.py --dataset coat --epochs 100 --batch_size 512 --recdim 64 --device cpu --test 1 --topk 10 --patience 100 --lr 0.001 --weight_decay 0.0001
python main.py --dataset movielensmini --epochs 100 --batch_size 512 --recdim 64 --device cpu --test 1 --topk 10 --patience 100 --lr 0.001 --weight_decay 0.0001
python main.py --dataset music4all --epochs 5 --batch_size 4096 --recdim 50 --device cpu --test 1 --topk 10 --patience 5 --ens_ratio 0.5 --lr 0.001 --weight_decay 0.0001
```

The paper’s Music4All pilot is documented in `experiments_music4all_pilot.md`, which includes the dataset construction, training commands, and validation/test results.

## Data and experiment notes
- The project supports public recommendation benchmarks and the Music4All pilot pipeline.
- User/item histories and preprocessing outputs are assumed to live under the repository’s `data/` directory.
- The training and evaluation flows are implemented end-to-end in `src/main.py`.
- The reproducibility package is intended to be mirrored to a public GitHub repository as part of the revision and camera-ready release workflow; the current workspace contains the complete artifact set for review and rerun.

## Citation
Please cite the manuscript when using this codebase in your research.


