# EMO-Disentanger
This is the official repository of ISMIR 2024 paper "Emotion-driven Piano Music Generation via Two-stage Disentanglement and Functional Representation".

[Demo page](https://emo-disentanger.github.io/) | [Model weights](https://drive.google.com/file/d/1eQoWuO-VzxtX-ZncQIoi87rqqcyKWeKz/view?usp=drive_link) | [Processed data](https://drive.google.com/file/d/1U9V5htFjgS9Cj59nJCCssbA7IXfagzwb/view?usp=drive_link)

## Environment
* **Python 3.8** and **CUDA 10.2** recommended
* Install dependencies (required)
```angular2html
pip install -r requirements.txt
```

* Install [fast transformer](https://github.com/idiap/fast-transformers) (required)
```
pip install --user pytorch-fast-transformers
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
1. Download and unzip [events](https://drive.google.com/file/d/1qvJfcXOftdKk3Bd0Lo1SaCsTltT_M3tu/view?usp=drive_link) and the [best weights](https://drive.google.com/file/d/1eQoWuO-VzxtX-ZncQIoi87rqqcyKWeKz/view?usp=drive_link) (make sure you're in repository root directory).
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
```angular2html
python3 stage2_accompaniment/inference.py \
        --configuration=stage2_accompaniment/config/emopia_finetune.yaml \
        --representation=functional \
        --inference_params=best_weight/Functional-two/emopia_acccompaniment_finetune/ep300_loss0.338_params.pt \
        --output_dir=generation/emopia_functional_two
```
4. To output synthesized audio together with midi files, add `--play_midi` in the commands.

### Other methods

1. For two-stage generation with REMI:
```angular2html
# stage1
python3 stage1_compose/inference.py \
        --configuration=stage1_compose/config/emopia_finetune.yaml \
        --representation=absolute \
        --mode=lead_sheet \
        --inference_params=best_weight/REMI-two/emopia_lead_sheet_finetune/ep016_loss0.846_params.pt \
        --output_dir=generation/emopia_remi_two

# stage2
python3 stage2_accompaniment/inference.py \
        --configuration=stage2_accompaniment/config/emopia_finetune.yaml \
        --representation=absolute \
        --inference_params=best_weight/REMI-two/emopia_acccompaniment_finetune/ep300_loss0.350_params.pt \
        --output_dir=generation/emopia_remi_two
```
2. For one-stage generation with REMI (baseline):
```angular2html
python3 stage1_compose/inference.py \
        --configuration=stage1_compose/config/emopia_finetune_full.yaml \
        --representation=absolute \
        --mode=full_song \
        --inference_params=best_weight/REMI-one/emopia_finetune/ep100_loss0.620_params.pt \
        --output_dir=generation/emopia_remi_one
```

## Train the model by yourself
Use **Two-stage generation with functional representation** as an example.
1. Use the provided [events](https://drive.google.com/file/d/1qvJfcXOftdKk3Bd0Lo1SaCsTltT_M3tu/view?usp=drive_link) directly or convert MIDI to events following the [steps](https://github.com/Yuer867/EMO-Disentanger/tree/main/representations#readme).
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
```angular2html
# pre-train on Pop1k7
python3 stage2_accompaniment/train.py \
        --configuration=stage2_accompaniment/config/pop1k7_pretrain.yaml \
        --representation=functional 

# finetune on EMOPIA (remember to add pretrained params in `emopia_finetune.yaml`)
python3 stage2_accompaniment/train.py \
        --configuration=stage2_accompaniment/config/emopia_finetune.yaml \
        --representation=functional
```

## Citation
If you find this project useful, please cite our paper:
```
@inproceedings{emodisentanger2024,
  author = {Jingyue Huang and Ke Chen and Yi-Hsuan Yang},
  title = {Emotion-driven Piano Music Generation via Two-stage Disentanglement and Functional Representation},
  booktitle={Proc. Int. Society for Music Information Retrieval Conf.},
  year = {2024}
}
```

