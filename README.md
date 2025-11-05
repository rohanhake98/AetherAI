<h1 align="center">AetherAI</h1>

<p align="center"><b>An open-source framework to evaluate, test and monitor ML and LLM-powered systems.</b></p>

<p align="center">
<a href="https://pypi.org/project/evidently/" target="_blank"><img src="https://img.shields.io/pypi/v/evidently" alt="PyPi"></a>
<a href="https://github.com/evidentlyai/evidently/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/github/license/evidentlyai/evidently" alt="License"></a>
<a href="https://pepy.tech/project/evidently" target="_blank"><img src="https://pepy.tech/badge/evidently" alt="PyPi Downloads"></a>
</p>

<p align="center">
  <img src="images/gh_header.png" alt="AetherAI Dashboard">
</p>

AetherAI is an open-source Python library for evaluating, testing, and monitoring machine learning models and LLM-powered systems in both development and production environments. It provides comprehensive tools for data quality assessment, model performance evaluation, data drift detection, and LLM evaluation.



# :bar_chart: What is AetherAI?

AetherAI is an open-source Python library to evaluate, test, and monitor ML and LLM systems—from experiments to production.

* 🔡 Works with tabular and text data.
* ✨ Supports evals for predictive and generative tasks, from classification to RAG.
* 📚 100+ built-in metrics from data drift detection to LLM judges.
* 🛠️ Python interface for custom metrics and descriptors.
* 🚦 Both offline evals and live monitoring.
* 💻 Open architecture: easily export data and integrate with existing tools.

AetherAI is very modular. You can start with one-off evaluations or host a full monitoring service with the built-in dashboard.

## 1. Reports and Test Suites

**Reports** compute and summarize various data, ML and LLM quality evals.
* Start with Presets and built-in metrics or customize.
* Best for experiments, exploratory analysis and debugging.
* View interactive Reports in Python or export as JSON, Python dictionary, HTML, or view in monitoring UI.

Turn any Report into a **Test Suite** by adding pass/fail conditions.
* Best for regression testing, CI/CD checks, or data validation.
* Zero setup option: auto-generate test conditions from the reference dataset.
* Simple syntax to set test conditions as `gt` (greater than), `lt` (less than), etc.

| Reports |
|--|
|![Report example](https://github.com/evidentlyai/docs/blob/eb1630cdd80d31d55921ff4d34fc7b5e6e9c9f90/images/concepts/report_test_preview.gif)|

## 2. Monitoring Dashboard

**Monitoring UI** service helps visualize metrics and test results over time.

You can choose:
* Self-host the open-source version. [Live demo](https://demo.evidentlyai.com).
* Sign up for [Evidently Cloud](https://www.evidentlyai.com/register) (Recommended).

Evidently Cloud offers a generous free tier and extra features like dataset and user management, alerting, and no-code evals. [Compare OSS vs Cloud](https://docs.evidentlyai.com/faq/oss_vs_cloud).

| Dashboard |
|--|
|![Dashboard example](https://github.com/evidentlyai/docs/blob/eb1630cdd80d31d55921ff4d34fc7b5e6e9c9f90/images/dashboard_llm_tabs.gif)|

# :woman_technologist: Install AetherAI

## Requirements

- Python 3.8 or higher
- Supported operating systems: Linux, macOS, Windows

## Installation

To install from PyPI:

```sh
pip install evidently
```

To install AetherAI using conda installer, run:

```sh
conda install -c conda-forge evidently
```

For development installation:

```sh
pip install -e .[dev,llm]
```

## Supported Integrations

AetherAI integrates with popular ML frameworks and tools:

- **ML Libraries**: scikit-learn, pandas, numpy, scipy
- **LLM Providers**: OpenAI, Anthropic, Hugging Face
- **Data Formats**: CSV, Parquet, JSON
- **Visualization**: Plotly for interactive reports
- **Storage**: SQL databases, cloud storage (S3, GCS)
- **Monitoring**: Prometheus, Grafana (via exporters)

# :arrow_forward: Getting started

## Reports

### LLM evals

> This is a simple Hello World. Check the Tutorials for more: [LLM evaluation](https://docs.evidentlyai.com/quickstart_llm).

Import the necessary components:

```python
import pandas as pd
from evidently import Report
from evidently import Dataset, DataDefinition
from evidently.descriptors import Sentiment, TextLength, Contains
from evidently.presets import TextEvals
```

Create a toy dataset with questions and answers.

```python
eval_df = pd.DataFrame([
    ["What is the capital of Japan?", "The capital of Japan is Tokyo."],
    ["Who painted the Mona Lisa?", "Leonardo da Vinci."],
    ["Can you write an essay?", "I'm sorry, but I can't assist with homework."]],
                       columns=["question", "answer"])
```

Create an Evidently Dataset object and add `descriptors`: row-level evaluators. We'll check for sentiment of each response, its length and whether it contains words indicative of denial.

```python
eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_df),
data_definition=DataDefinition(),
descriptors=[
    Sentiment("answer", alias="Sentiment"),
    TextLength("answer", alias="Length"),
    Contains("answer", items=['sorry', 'apologize'], mode="any", alias="Denials")
])
```

You can view the dataframe with added scores:

```python
eval_dataset.as_dataframe()
```

To get a summary Report to see the distribution of scores:

```python
report = Report([
    TextEvals()
])

my_eval = report.run(eval_dataset)
my_eval
# my_eval.json()
# my_eval.dict()
```
You can also choose other evaluators, including LLM-as-a-judge and configure pass/fail conditions.

### Data and ML evals

> This is a simple Hello World. Check the Tutorials for more: [Tabular data](https://docs.evidentlyai.com/quickstart_ml).

Import the Report, evaluation Preset and toy tabular dataset.

```python
import pandas as pd
from sklearn import datasets

from evidently import Report
from evidently.presets import DataDriftPreset

iris_data = datasets.load_iris(as_frame=True)
iris_frame = iris_data.frame
```

Run the **Data Drift** evaluation preset that will test for shift in column distributions. Take the first 60 rows of the dataframe as "current" data and the following as reference.  Get the output in Jupyter notebook:

```python
report = Report([
    DataDriftPreset(method="psi")
],
include_tests="True")
my_eval = report.run(iris_frame.iloc[:60], iris_frame.iloc[60:])
my_eval
```

You can also save an HTML file. You'll need to open it from the destination folder.

```python
my_eval.save_html("file.html")
```

To get the output as JSON or Python dictionary:
```python
my_eval.json()
# my_eval.dict()
```
You can choose other Presets, create Reports from indiviudal Metrics and configure pass/fail conditions.

## Monitoring dashboard

This launches a demo project in the locally hosted Evidently UI. Sign up for [Evidently Cloud](https://docs.evidentlyai.com/docs/setup/cloud) to instantly get a managed version with additional features.


If you have [uv](https://docs.astral.sh/uv/) you can run AetherAI UI with a single command.
```shell
uv run --with evidently aetherai ui --demo-projects all
```

If you haven't, create a virtual environment using the standard approach.
```
pip install virtualenv
virtualenv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

After installing AetherAI (`pip install evidently`), run the AetherAI UI with the demo projects:
```
aetherai ui --demo-projects all
```

Visit **localhost:8000** to access the UI.

## 📚 Examples

AetherAI comes with comprehensive examples demonstrating various use cases:

- **Data Drift Detection**: Monitor changes in data distribution over time
- **LLM Evaluation**: Evaluate language model outputs using various metrics
- **Classification Performance**: Assess model accuracy, precision, recall, and other metrics
- **Regression Analysis**: Evaluate regression model performance with MAE, RMSE, etc.
- **Recommendation Systems**: Analyze ranking quality and recommendation diversity
- **Text Analysis**: Use descriptors to analyze text properties like sentiment, length, and content

Check out the [examples](examples/) directory for detailed notebooks and tutorials.

## 🖥️ Command Line Interface

AetherAI provides a powerful CLI for running evaluations and starting the monitoring dashboard:

```bash
# Start the monitoring UI with demo projects
aetherai ui --demo-projects all

# Generate a specific demo project
aetherai ui --demo-projects data_drift

# Run with custom configuration
aetherai ui --workspace ./my_workspace --port 8080

# Run a report from command line (coming soon)
aetherai report --config my_report.yaml
```

# 🚦 What can you evaluate?

AetherAI has 100+ built-in evals. You can also add custom ones.

Here are examples of things you can check:

|                           |                          |
|:-------------------------:|:------------------------:|
| **🔡 Text descriptors**   | **📝 LLM outputs**       |
| Length, sentiment, toxicity, language, special symbols, regular expression matches, etc. | Semantic similarity, retrieval relevance, summarization quality, etc. with model- and LLM-based evals. |
| **🛢 Data quality**       | **📊 Data distribution drift** |
| Missing values, duplicates, min-max ranges, new categorical values, correlations, etc. | 20+ statistical tests and distance metrics to compare shifts in data distribution. |
| **🎯 Classification**     | **📈 Regression**        |
| Accuracy, precision, recall, ROC AUC, confusion matrix, bias, etc. | MAE, ME, RMSE, error distribution, error normality, error bias, etc. |
| **🗂 Ranking (inc. RAG)** | **🛒 Recommendations**   |
| NDCG, MAP, MRR, Hit Rate, etc. | Serendipity, novelty, diversity, popularity bias, etc. |

## 🏗️ Architecture

AetherAI follows a modular architecture that makes it easy to integrate into your ML workflow:

1. **Datasets**: Load and preprocess your data with built-in support for various data types
2. **Metrics**: Compute metrics on your data using built-in or custom functions
3. **Reports**: Generate detailed reports with visualizations and insights
4. **Tests**: Convert reports to automated tests with pass/fail conditions
5. **Dashboard**: Monitor metrics over time with the built-in web UI

## 🔧 Key Features

AetherAI provides a comprehensive set of tools for ML and LLM evaluation:

- **Comprehensive Metrics**: Over 100 built-in metrics covering data quality, model performance, and business metrics
- **LLM Evaluation**: Specialized tools for evaluating language models including toxicity, sentiment, and custom LLM judges
- **Data Drift Detection**: 20+ statistical tests to monitor data distribution shifts
- **Interactive Reports**: Generate detailed reports in Jupyter notebooks, HTML, or JSON formats
- **Monitoring Dashboard**: Real-time visualization of metrics and test results
- **Custom Descriptors**: Create custom text analysis functions with the descriptor system
- **Presets**: Pre-built evaluation templates for common use cases
- **Test Suites**: Convert reports to automated tests with pass/fail conditions

## 🤖 LLM Evaluation

AetherAI provides specialized tools for evaluating Large Language Models:

- **Text Descriptors**: Analyze text properties like sentiment, length, toxicity, and custom patterns
- **LLM-as-a-Judge**: Use LLMs to evaluate other LLM outputs with customizable prompts
- **RAG Evaluation**: Assess retrieval-augmented generation systems
- **Bias Detection**: Identify potential biases in LLM outputs
- **Toxicity Analysis**: Detect harmful or inappropriate content
- **Custom Evaluations**: Create your own LLM evaluation criteria

## 🐍 Python API

AetherAI provides a simple Python API for integration into your ML workflows:

```python
import pandas as pd
from aetherai import Report, Dataset
from aetherai.presets import DataDriftPreset

# Load your data
df = pd.read_csv('your_data.csv')

# Create a report
report = Report(metrics=[DataDriftPreset()])

# Run the evaluation
result = report.run(df)

# View results
result.show()
```

## 🤝 Community and Support

- Join our [Discord community](https://discord.gg/xZjKRaNp8b) for discussions and support
- Check out our [documentation](https://docs.evidentlyai.com) for detailed guides
- Follow us on [Twitter](https://twitter.com/evidentlyai) for updates
- Report issues on our [GitHub repository](https://github.com/evidentlyai/evidently/issues)

## 📄 License

AetherAI is licensed under the [Apache 2.0 License](LICENSE).

## 🙏 Acknowledgements

AetherAI is built on top of the excellent work done by the Evidently AI team and community. We're grateful for their contributions to the open-source ML evaluation ecosystem.

# :computer: Contributions

We welcome contributions! Read the [Guide](CONTRIBUTING.md) to learn more.

If you're interested in contributing, here are some ways you can help:

- Report bugs or request features
- Improve documentation
- Submit pull requests with bug fixes or new features
- Share your use cases and examples
- Help translate documentation

Before contributing, please read our [Contributing Guide](CONTRIBUTING.md).

