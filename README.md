# ParsiEval: A Benchmark for Persian Language Understanding

This project introduces **ParsiEval**, a comprehensive benchmark designed to evaluate the performance of Large Language Models (LLMs) on Persian language tasks. The primary goal of ParsiEval is to provide a standardized and challenging testbed for assessing the capabilities of LLMs in understanding and reasoning in Persian.

## Dataset

The ParsiEval dataset consists of 364 multiple-choice questions that span a wide variety of topics, including:

- Science
- History
- Literature
- General Knowledge

Each question is designed to test a model's ability to comprehend context, recall facts, and make logical inferences in Persian.

## Goal and Motivation

The development of high-quality benchmarks for languages other than English is crucial for advancing the field of multilingual NLP. ParsiEval aims to fill this gap for the Persian language by providing a robust evaluation suite that can be used to:

- Track the progress of Persian language models over time.
- Identify the strengths and weaknesses of different model architectures.
- Drive research and development in Persian language technology.

## Results

Here are the results of the evaluation for different models.

### API-Based Models

These models were evaluated via their respective APIs.

| Model               | Accuracy | Correct Answers | Total Questions | Avg. Latency | Total Latency |
| ------------------- | -------- | --------------- | --------------- | ------------ | ------------- |
| openai/gpt-oss-120b | 70.05%   | 255             | 364             | 2.84s        | 1034.47s      |
| openai/gpt-oss-20b  | 63.74%   | 232             | 364             | 2.73s        | 994.36s       |

### Locally-Run Models

These models were run on a local machine.

| Model                             | Accuracy | Correct Answers | Total Questions | Avg. Latency | Total Latency |
| --------------------------------- | -------- | --------------- | --------------- | ------------ | ------------- |
| gemma3:1b                         | 23.08%   | 84              | 364             | 0.19s        | 70.21s        |
| gemma3:4b                         | 43.13%   | 157             | 364             | 0.13s        | 46.63s        |
| hf.co/mshojaei77/gemma-3-4b-persian | 43.96%   | 157             | 364             | 0.21s        | 77.41s        |
| gemma2:2b                         | 15.93%   | 58              | 364             | 0.81s        | 295.59s       |
| llama3.2:3b                       | 34.34%   | 125             | 364             | 0.13s        | 46.11s        |
| gemma-2-2b-fa-v2                  | 29.40%   | 107             | 364             | 0.43s        | 156.58s       |
| EXAONE-3.5-2.4B                   | 25.55%   | 93              | 364             | 0.89s        | 325.25s       |
| llama3.2:1b                       | 17.03%   | 62              | 364             | 0.17s        | 63.05s        |
| qwen2.5:0.5b                      | 19.78%   | 72              | 364             | 1.00s        | 365.51s       |
| qwen2.5:1.5b                      | 32.14%   | 117             | 364             | 0.07s        | 24.79s        |
| qwen2.5-coder:0.5b                | 23.63%   | 86              | 364             | 0.22s        | 78.65s        |
| phi4-mini                         | 29.67%   | 108             | 364             | 0.24s        | 88.75s        |