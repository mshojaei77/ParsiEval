# ParsiEval: A Benchmark for Persian Language Understanding

This project introduces **ParsiEval**, a comprehensive benchmark designed to evaluate the performance of Large Language Models (LLMs) on Persian language tasks. The primary goal of ParsiEval is to provide a standardized and challenging testbed for assessing the capabilities of LLMs in understanding and reasoning in Persian.

## Dataset

The ParsiEval dataset consists of 364 multiple-choice questions that span a wide variety of topics, including:


- History
- Literature
- General Knowledge
- Science

Each question is designed to test a model's ability to comprehend context, recall facts, and make logical inferences in Persian.

## Goal and Motivation

The development of high-quality benchmarks for languages other than English is crucial for advancing the field of multilingual NLP. ParsiEval aims to fill this gap for the Persian language by providing a robust evaluation suite that can be used to:

- Track the progress of Persian language models over time.
- Identify the strengths and weaknesses of different model architectures.
- Drive research and development in Persian language technology.

## Results

Here are the results of the evaluation for different models.

### Accuracy

#### Top Models
Analysis of the highest performing models

![Accuracy of API-Based Models](plots/accuracy_top_models.png)

#### Edge-Device Models
Examination of smaller models suitable for edge devices

![Accuracy of Edge-Device Models](plots/accuracy_edge_models.png)

### Accuracy vs Latency

#### Top Models
Evaluation of the trade-off between accuracy and response time

![Accuracy vs. Latency for API-Based Models](plots/accuracy_vs_latency_top_models.png)

#### Edge-Device Models
Analysis of speed-performance balance in edge models

![Accuracy vs. Latency for Edge-Device Models](plots/accuracy_vs_latency_edge_models.png)

### Accuracy vs Parameters

#### Top Models
Investigation of the relationship between model size and performance

![Accuracy vs. Parameters for API-Based Models](plots/accuracy_vs_parameters_top_models.png)

#### Edge-Device Models
Assessment of performance scaling in small models (<4B parameters)

![Accuracy vs. Parameters for Edge-Device Models](plots/accuracy_vs_parameters_edge_models.png)
