# TaiPhoneEval: LLM Evaluation for Traditional Chinese

A comprehensive evaluation  designed to assess LLMs, with specialized support for Traditional Chinese language evaluation.

## Overview

TaiPhone provides a robust suite of evaluation tools and scripts for:
- Multiple-choice question (MCQ) assessment
- MT-Bench evaluation in Traditional Chinese
- Flexible LLM backend integration via LiteLLM
- High-performance inference with VLLM

## Quick Start

1. **Installation**
   ```bash
   poetry install
   ```

2. **Configuration**
   ```bash
   # Set your OpenAI API key (if needed)
   export OPENAI_API_KEY=sk-your_api_key
   ```

## Project Architecture

```
.
├── bash/           # Automation and utility scripts
├── dict/           # Reference data and dictionaries
├── result/         # Evaluation outputs and metrics
├── scripts/        # Core evaluation modules
│   ├── eval_mcq.py           # MCQ evaluation engine
│   └── eval_mt_bench_zhtw.py # Traditional Chinese MT-Bench evaluator
├── pyproject.toml  # Project configuration
└── poetry.lock     # Dependency lock file
```

## Usage Guide

### 1. Launch VLLM Server
```bash
vllm serve $MODEL_NAME --max-model-len 8192
```

### 2. Run Evaluations

#### Multiple-Choice Question Evaluation

**Taiwan Culture MCQ Dataset**
```bash
python scripts/eval_mcq.py \
    --dataset aqweteddy/Taiwan-Curlture-MCQ \
    --url http://localhost:8000/v1 \
    --output_path result/mcq/$MODEL_NAME.json \
    --lang zh
```

**MMLU-Redux Dataset**
```bash
python scripts/eval_mcq.py \
    --dataset aqweteddy/MMLU-Redux-MCQ \
    --url http://localhost:8000/v1 \
    --output_path result/mcq/$MODEL_NAME.json \
    --lang en
```

#### Traditional Chinese MT-Bench Evaluation
```bash
python scripts/eval_mt_bench_zhtw.py \
    --dataset ZoneTwelve/mt-bench-tw \
    --url http://localhost:8000/v1 \
    --output_path result/mt-bench-zhtw/$MODEL_NAME.json
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{taiphone-eval,
  author = {aqweteddy},
  title = {TaiPhone: LLM Evaluation Framework for Traditional Chinese},
  year = {2025},
  url = {https://github.com/aqweteddy/taiphone-eval}
}
```
