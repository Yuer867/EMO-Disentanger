# Data Processing

## Convert MIDI to events

### EMOPIA
1. Download and unzip [processed midi data](https://drive.google.com/file/d/1U9V5htFjgS9Cj59nJCCssbA7IXfagzwb/view?usp=drive_link) in the repository root directory.
2. Three conversion options are provided for different generation stages with functional representation (`--representation=functional`) and REMI (`--representation=absolute`).
```angular2html
# functional representation for lead sheet generation
python3 representations/midi2events_emopia.py --representation=functional --event_type=lead

# functional representation for performance generation conditioned on lead sheet
python3 representations/midi2events_emopia.py --representation=functional --event_type=lead2full

# functional representation for performance generation from scratch
python3 representations/midi2events_emopia.py --representation=functional --event_type=full
```

### HookTheory
1. Download [JSON](https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz) data released in [SheetSage](https://github.com/chrisdonahue/sheetsage) and put it into `midi_data/HookTheory/`.
2. Convert MIDI to lead sheet using functional representation. For REMI, set `--representation=absolute`.
```angular2html
# functional representation for lead sheet generation
python3 representations/midi2events_hooktheory.py --representation=functional 
```

### Pop1k7
1. Two conversion options are provided for different generation stages with functional representation (`--representation=functional`) and REMI (`--representation=absolute`).
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
python3 representations/events2words.py --representation=absolute
```

## Data splits
For EMOPIA and HookTheory, run the following command. For Pop1k7, please refer to [Compose & Embellish](https://github.com/slSeanWU/Compose_and_Embellish).
```angular2html
python3 representations/data_splits.py
```

