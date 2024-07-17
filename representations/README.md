# Data Processing

## Convert MIDI to events

### EMOPIA
1. Download and unzip [processed midi data](https://drive.google.com/file/d/1j76WMK7jjORWeRBGRM8zTd6H1Iand6CZ/view?usp=sharing) (make sure you're in repository root directory).
2. Convert MIDI to events.
```angular2html
# REMI representation
python3 representations/midi2events_emopia.py --representation=absolute

# REMI representation (transpose to C major / c minor based on original keymode)
python3 representations/midi2events_emopia.py --representation=transpose

# REMI representation (transpose to C major / c minor based on emotion)
python3 representations/midi2events_emopia.py --representation=transpose_rule

# functional representation (ablated)
python3 representations/midi2events_emopia.py --representation=ablated

# functional representation
python3 representations/midi2events_emopia.py --representation=functional
```

### HookTheory

1. Download [JSON](https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz) data released in [SheetSage](https://github.com/chrisdonahue/sheetsage) and put it into `midi_data/HookTheory/`.
2. Convert MIDI to events.
```angular2html
# functional representation (others are similar to EMOPIA)
python3 representations/midi2events_hooktheory.py --representation=functional
```

## Build Vocabulary
```angular2html
# functional representation (others are similar to EMOPIA)
python3 representations/events2words.py --representation=functional
```

## Data splits
```angular2html
python3 representations/data_splits.py
```
