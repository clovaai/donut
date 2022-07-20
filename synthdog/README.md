# SynthDoG 🐶: Synthetic Document Generator

SynthDoG is synthetic document generator for visual document understanding (VDU).

![image](../misc/sample_synthdog.png)

## Prerequisites

- python>=3.6
- [synthtiger](https://github.com/clovaai/synthtiger) (`pip install synthtiger`)

## Usage

```bash
# Set environment variable (for macOS)
$ export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

synthtiger -o {dataset_path}/SynthDoG_en -c 100 -w 4 -v template.py SynthDog config_en.yaml

{'config': 'config_en.yaml',
 'count': 100,
 'name': 'SynthDog',
 'output': 'outputs/SynthDoG_en',
 'script': 'template.py',
 'verbose': True,
 'worker': 4}
{'aspect_ratio': [1, 2],
     .
     .
 'quality': [50, 95],
 'short_size': [720, 1024]}
Generated 1 data
Generated 2 data
Generated 3 data
     .
     .
Generated 99 data
Generated 100 data
108.74 seconds elapsed
```

To generate ECJK samples:
```bash
# english
synthtiger -o {dataset_path}/synthdog-en -w 4 -v template.py SynthDoG config_en.yaml

# chinese
synthtiger -o {dataset_path}/synthdog-zh -w 4 -v template.py SynthDoG config_zh.yaml

# japanese
synthtiger -o {dataset_path}/synthdog-ja -w 4 -v template.py SynthDoG config_ja.yaml

# korean
synthtiger -o {dataset_path}/synthdog-ko -w 4 -v template.py SynthDoG config_ko.yaml
```
