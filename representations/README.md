# Data Processing

## Convert MIDI to events

### EMOPIA
1. Create folder `midi_data` in the root directory.
2. Download and unzip [EMOPIA+](https://zenodo.org/records/13122742) in the `midi_data` folder.
3. Three conversion types are needed for different generation stages with functional representation (`--representation=functional`) or REMI (`--representation=remi`).
```angular2html
# functional representation for lead sheet generation
python3 representations/midi2events_emopia.py --representation=functional --event_type=lead

# functional representation for performance generation conditioned on lead sheet
python3 representations/midi2events_emopia.py --representation=functional --event_type=lead2full

# functional representation for performance generation from scratch
python3 representations/midi2events_emopia.py --representation=functional --event_type=full
```

### HookTheory
1. Create folder `HookTheory` in the `midi_data` folder.
2. Download [JSON](https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz) data released in [SheetSage](https://github.com/chrisdonahue/sheetsage) and put it into `midi_data/HookTheory/`. 
3. Convert MIDI to lead sheet with functional representation (`--representation=functional`) or REMI (`--representation=remi`)
```angular2html
# functional representation for lead sheet generation
python3 representations/midi2events_hooktheory.py --representation=functional 
```

### Pop1k7
1. Download and unzip [Pop1K7-emo](https://zenodo.org/records/13167761) in the `midi_data` folder.
2. Put processed dataset [pop1k7_leedsheet2midi](https://huggingface.co/slseanwu/compose-and-embellish-pop1k7/tree/main/datasets/stage02_embellish) provided by [Compose & Embellish](https://github.com/slSeanWU/Compose_and_Embellish) into `midi_data/Pop1K7-emo`.
1. Two conversion types are needed for different generation stages with functional representation (`--representation=functional`) or REMI (`--representation=remi`).
```angular2html
# functional representation for performance generation conditioned on lead sheet
python3 representations/midi2events_pop1k7.py --representation=functional --event_type=lead2full

# functional representation for performance generation from scratch
python3 representations/midi2events_pop1k7.py --representation=functional --event_type=full
```

## Build Vocabulary
```angular2html
# functional representation
python3 representations/events2words.py --representation=functional

# REMI representation
python3 representations/events2words.py --representation=remi
```

## Data splits
For EMOPIA and HookTheory, run the following command. For Pop1k7, please refer to [Compose & Embellish](https://github.com/slSeanWU/Compose_and_Embellish).
```angular2html
python3 representations/data_splits.py
```

