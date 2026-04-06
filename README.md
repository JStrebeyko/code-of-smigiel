# Code of ŚMIGIEL

## Overview

This repository shows the steps involved in creation of [ŚMIGIEL Dataset](HF), used in [PolEval 2025 Task 1: Spotting Machine-Generated Text from LLMs for Polish](https://github.com/poleval/2025-smigiel).

Specifically, the code of this repo can

- acquire source datasets of texts from various domains in Polish
- process text, inc. cleaning, filtration, selection and prefix extraction
- prepare domain-specific prompts
- setup execution environment in a cluster for Multi-GPU processing
- manage LLMs, inc. inference using various decoding strategies
- group, balance and sample the data
- perform customizable cleaning on the generated data

The code in the repository can be divided into two parts, according to the environment in which they are intended to be run.

1. the CLI, offering the basic data transformation steps. It covers all the data preparation and postprocessing steps, along with utilities for balancing the domains, filtering the content and composing the resulting dataset. Intended to be run locally.
2. the inference script, focusing on cluster configuration, LLM management and generation. It is driven by dynamically composed Slurm configuration file and is intended for the cluster.

## Quickstart

The project uses [uv](https://docs.astral.sh/uv/). Upon install, run the following in the root directory to install the dependencies:

```shell
uv sync
```

To run the CLI, use

```shell
uv run cli.py [command] [...options]
```

```shell
uv run cli.py preprocess --datasets wiki
```

## About the Shared Task

[Śmigiel](https://github.com/poleval/2025-smigiel) is a Shared Task on Machine-Generated-Text (MGT) Detection, which for the first time took place as part of [PolEval 2025](http://poleval.pl/tasks/task1) competition organized by [The Linguistic Engineering (LE) Group](https://zil.ipipan.waw.pl/), part of the Department of Artificial Intelligence at the Institute of Computer Science, Polish Academy of Sciences (IPI PAN) <!-- TODO: add a sentence on the participants? -->

## The procedure

ŚMIGIEL data is built upon 12 datasets of human-written texts in Polish. Each of them is processed in a roughly the same way:
[I wish I had a flowchart here]

raw -> preprocessing -> prefix extraction and prompt creation -> balancing within domains -> LLM completion by each of the models -> postprocessing of the ouput, inc. final dataset composition

TBC
