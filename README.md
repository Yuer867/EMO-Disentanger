# EMO-Disentanger
This is the official repository of ISMIR 2024 paper "Emotion-driven Piano Music Generation via Two-stage Disentanglement and Functional Representation".

[Paper](https://arxiv.org/abs/2407.20955) | [Demo page](https://emo-disentanger.github.io/) | [Model weights](https://drive.google.com/file/d/15Gc8PWbkoOeXCTrpDMKsgptoL17u49QG/view?usp=sharing) | [Dataset: EMOPIA+](https://zenodo.org/records/13122742) | [Dataset: Pop1K7 & Pop1K7-emo](https://zenodo.org/records/13167761)

## Environment
* **Python 3.8** and **CUDA 10.2** recommended
* Install dependencies (required)
```angular2html
pip install -r requirements.txt
```

* For stage2, install [fast transformer](https://github.com/idiap/fast-transformers) or transformers (required)
```
# fast-transformers (the package used in the paper, but may not work in some cuda versions)
pip install --user pytorch-fast-transformers

# transformers
pip install transformers==4.28.0
```

* Install [midi2audio](https://github.com/bzamecnik/midi2audio) to synthesize generated MIDI to audio (optional)
```
pip install midi2audio
wget https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3+20200602.tar.xz
tar -xzvf SalamanderGrandPiano-SF2-V3+20200602.tar.xz
```

## Quick Start

### Emotion-driven piano performance generation (with our trained models)
**Method: Two-stage generation with functional representation**
1. Download and unzip [events](https://drive.google.com/file/d/1NRisx-FpjcsXov1jmtrbAhtFBFBkGRgW/view?usp=sharing) and the [best weights](https://drive.google.com/file/d/15Gc8PWbkoOeXCTrpDMKsgptoL17u49QG/view?usp=sharing) (make sure you're in repository root directory).
2. Stage1: Generate lead sheet with **Positive** or **Negative** emotion conditions (i.e., Valence Modeling).
```angular2html
python3 stage1_compose/inference.py \
        --configuration=stage1_compose/config/emopia_finetune.yaml \
        --representation=functional \
        --mode=lead_sheet \
        --inference_params=best_weight/Functional-two/emopia_lead_sheet_finetune/ep016_loss0.685_params.pt \
        --output_dir=generation/emopia_functional_two
```
3. Stage2: Generate music performance based on generated lead sheet in stage1 to convey **4Q** emotions (i.e., Arousal Modeling).
* (Option 1) with Performer backbone (install fast-transformers)
```angular2html
python3 stage2_accompaniment/inference.py \
        --model_type=performer \
        --configuration=stage2_accompaniment/config/emopia_finetune.yaml \
        --representation=functional \
        --inference_params=best_weight/Functional-two/emopia_acccompaniment_finetune/ep300_loss0.338_params.pt \
        --output_dir=generation/emopia_functional_two
```
* (Option 2) with GPT-2 backbone (install transformers)
```angular2html
python3 stage2_accompaniment/inference.py \
        --model_type=gpt2 \
        --configuration=stage2_accompaniment/config/emopia_finetune_gpt2.yaml \
        --representation=functional \
        --inference_params=best_weight/Functional-two/emopia_acccompaniment_finetune_gpt2/ep300_loss0.120_params.pt \
        --output_dir=generation/emopia_functional_two
```
4. To output synthesized audio together with midi files, add `--play_midi` in the commands.

### Other methods

1. For two-stage generation with REMI:
```angular2html
# stage1
python3 stage1_compose/inference.py \
        --configuration=stage1_compose/config/emopia_finetune.yaml \
        --representation=remi \
        --mode=lead_sheet \
        --inference_params=best_weight/REMI-two/emopia_lead_sheet_finetune/ep016_loss0.846_params.pt \
        --output_dir=generation/emopia_remi_two

# stage2
# (Option 1) with Performer backbone (install fast-transformers)
python3 stage2_accompaniment/inference.py \
        --model_type=performer \
        --configuration=stage2_accompaniment/config/emopia_finetune.yaml \
        --representation=remi \
        --inference_params=best_weight/REMI-two/emopia_acccompaniment_finetune/ep300_loss0.350_params.pt \
        --output_dir=generation/emopia_remi_two

# (Option 2) with GPT-2 backbone (install transformers)
python3 stage2_accompaniment/inference.py \
        --model_type=gpt2 \
        --configuration=stage2_accompaniment/config/emopia_finetune_gpt2.yaml \
        --representation=remi \
        --inference_params=best_weight/REMI-two/emopia_acccompaniment_finetune_gpt2/ep300_loss0.136_params.pt \
        --output_dir=generation/emopia_remi_two
```
2. For one-stage generation with REMI (baseline):
```angular2html
python3 stage1_compose/inference.py \
        --configuration=stage1_compose/config/emopia_finetune_full.yaml \
        --representation=remi \
        --mode=full_song \
        --inference_params=best_weight/REMI-one/emopia_finetune/ep100_loss0.620_params.pt \
        --output_dir=generation/emopia_remi_one
```

## Train the model by yourself
Use **Two-stage generation with functional representation** as an example.
1. Use the provided [events](https://drive.google.com/file/d/1NRisx-FpjcsXov1jmtrbAhtFBFBkGRgW/view?usp=sharing) directly or convert MIDI to events following the [steps](https://github.com/Yuer867/EMO-Disentanger/tree/main/representations#readme).
2. Stage1: Valence Modeling (lead sheet generation)
```angular2html
# pre-train on HookTheory
python3 stage1_compose/train.py \
        --configuration=stage1_compose/config/hooktheory_pretrain.yaml \
        --representation=functional

# finetune on EMOPIA (remember to add pretrained params in `emopia_finetune.yaml`)
python3 stage1_compose/train.py \
        --configuration=stage1_compose/config/emopia_finetune.yaml \
        --representation=functional
```
3. Stage2: Arousal Modeling (performance generation)
* (Option 1) with Performer backbone (install fast-transformers)
```angular2html
# pre-train on Pop1k7
python3 stage2_accompaniment/train.py \
        --model_type=performer \
        --configuration=stage2_accompaniment/config/pop1k7_pretrain.yaml \
        --representation=functional 

# finetune on EMOPIA (remember to add pretrained params in `emopia_finetune.yaml`)
python3 stage2_accompaniment/train.py \
        --model_type=performer \
        --configuration=stage2_accompaniment/config/emopia_finetune.yaml \
        --representation=functional
```
* (Option 2) with GPT-2 backbone (install transformers)
```angular2html
# pre-train on Pop1k7
python3 stage2_accompaniment/train.py \
        --model_type=gpt2 \
        --configuration=stage2_accompaniment/config/pop1k7_pretrain_gpt2.yaml \
        --representation=functional 

# finetune on EMOPIA (remember to add pretrained params in `emopia_finetune_gpt2.yaml`)
python3 stage2_accompaniment/train.py \
        --model_type=gpt2 \
        --configuration=stage2_accompaniment/config/emopia_finetune_gpt2.yaml \
        --representation=functional
```

## Dataset
We open source the processed midi data as follows:
* [EMOPIA+](https://zenodo.org/records/13122742) for fine-tuning both stages, derived from emotion-annotated multi-modal dataset [EMOPIA](https://arxiv.org/abs/2108.01374). 
  * We applied [Midi_Toolkit](https://github.com/RetroCirce/Midi_Toolkit) for melody extraction and [link](https://github.com/Dsqvival/hierarchical-structure-analysis/tree/main/preprocessing/exported_midi_chord_recognition) for chord recognition to extract lead sheet from piano performance. 
  * To refine key signatures, we applied both MIDI-based ([Midi toolbox](https://github.com/miditoolbox/)) and audio-based ([madmom](https://github.com/CPJKU/madmom)) key detection methods and manually corrected the clips where the two methods disagreed.
* [Pop1K7-emo](https://zenodo.org/records/13167761) for pretraining second stage, derived from piano performance dataset [AILabs.tw Pop1K7](https://github.com/YatingMusic/compound-word-transformer).
  * Please refer to [Compound Work Transformer](https://arxiv.org/abs/2101.02402) for lead sheet extraction algorithms.
  * Key signatures are detected using [Midi toolbox](https://github.com/miditoolbox/).

## Citation
If you find this project useful, please cite our paper:
```
@inproceedings{emodisentanger2024,
  author = {Jingyue Huang and Ke Chen and Yi-Hsuan Yang},
  title = {Emotion-driven Piano Music Generation via Two-stage Disentanglement and Functional Representation},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference, {ISMIR}},
  year = {2024}
}
```

