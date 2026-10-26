# AlignFSM: Behavioral Alignment of Legal Contracts and Smart Contracts via Finite State Machines

This repository contains the source code, benchmark, and evaluation scripts for **AlignFSM**, a multi-agent framework that bridges the semantic gap between natural-language legal contracts and executable smart contracts. AlignFSM uses finite state machines (FSMs) as an intermediate representation to enforce deontic logic, temporal constraints, and behavioral alignment, and iteratively repairs the generated code against the extracted contract specification.

## Framework Overview

AlignFSM consists of four collaborating agents:



1. **Agent 1 — Contract-to-FSM Extraction** (`agent1_contractFSM.py`): extracts a deontic FSM (states, transitions, rights, obligations, permissions, conditions, and temporal constraints) from a legal contract.

2. **Agent 2 — FSM-to-Code Generation** (`agent2_codegen.py`): generates a Solidity smart contract from the extracted FSM, including state guards and transition functions.

3. **Agent 3 — Code-to-FSM Reverse Extraction** (`agent3_codeFSM.py`): reconstructs an FSM from the generated Solidity code.

4. **Agent 4 — FSM Alignment and Repair** (`agent4_alignFSM.py`): compares the reverse-extracted FSM with the original contract FSM and feeds alignment feedback to Agent 2 to regenerate the code.

The end-to-end pipeline is orchestrated by `pipeline.py`.

## Repository Structure



```
.
├── pipeline.py               # Main end-to-end pipeline orchestrator
├── agent1_contractFSM.py     # Agent 1: contract -> FSM
├── agent2_codegen.py         # Agent 2: FSM -> Solidity
├── agent3_codeFSM.py         # Agent 3: Solidity -> FSM
├── agent4_alignFSM.py        # Agent 4: FSM alignment and repair
├── evaluate.py               # Evaluation entry point
├── baseline/
│   ├── prompts/              # BPMN/CML/SPESC/Symboleo prompt templates
│   ├── grammar/              # CML/SPESC/Symboleo grammar definitions
│   └── bpmn_fewshot/         # BPMN few-shot exemplars
├── datasets/
│   ├── CUAD/CUAD_dataset.jsonl
│   └── MAUD/contracts/       # Raw MAUD contract texts
└── Outputs/                  # Generated FSMs and Solidity outputs
```

## Running the Pipeline

Configure the run at the top of `pipeline.py`:



* `DATASET_PATH`: path to the preprocessed contract file (one JSON object per line, containing the contract text and identifier).

* `MODEL_NAMES`: target backend model(s), e.g. `gpt-4o-mini`, `deepseek-chat`, or `gemini-2.5-flash-lite`.

* `RUN_ONE_SAMPLE` / `SAMPLE_START_INDEX`: run a single contract for testing.

* `MAX_ITER`: maximum Agent 2/Agent 4 repair iterations.

Then run:



```
python pipeline.py
```

The generated FSMs, Solidity code, and alignment feedback are written under `Outputs/`.

## Evaluation

`evaluate.py` extracts deontic behaviors from both the contract text and the generated code and computes behavioral fitness, precision, and the final alignment score.



```
python evaluate.py --generated_dir <path-to-generated-outputs> \
                   --contracts_jsonl <path-to-contracts.jsonl> \
                   --output_csv metrics.csv \
                   --comparison_csv method_comparison.csv
```

Key options:



* `--ranking_metric {BehavioralAlignmentIndex,FinalScore}`: metric used for per-contract method ranking.

* `--extractor {improved,legacy}`: behavior extraction backend.

* `--methods`, `--models`, `--contract_ids`: comma-separated filters.

* `--batch_size` / `--batch_index`: evaluate in batches.

* `--disable_bloat_penalty`, `--disable_structure_score`, `--sanitize_baseline_code`, `--strict_baseline_full`: scoring variants.

Per-contract metrics are written to the `--output_csv` file, and per-contract method rankings to the `--comparison_csv` file.

## Baselines

The `baseline/` directory contains the resources used by the BPMN, CML, SPESC, and Symboleo baselines:



* `prompts/`: prompt templates for each baseline specification.

* `grammar/`: grammar definitions (`cml.xtext`, `spesc.xtext`, `symboleo.xtext`).

* `bpmn_fewshot/`: few-shot exemplars for the BPMN baseline.

## Datasets



* **CUAD**: `datasets/CUAD/CUAD_dataset.jsonl`.

* **MAUD**: raw contract texts under `datasets/MAUD/contracts/`.

## Notes on Reproducibility

LLM outputs are inherently nondeterministic; the experiments use temperature 0.0. Minor numerical differences between re-runs and the reported results are expected.