# AI-Powered Finance Analyst

This project implements an AI-powered finance analyst using the Perplexity API. Follow the steps below to set up and run the project.

This project is an Agentic Workflow, which combines data from Yahoo Finance and DuckDuckGo to generate a report on a given stock
---

## Setup Instructions

### Step 1: Set up your virtual environment

Create and activate a virtual environment:

```bash
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
```

### Step 2: Export your Perplexity API key

Set your Perplexity API key as an environment variable or add it in .env file

```bash
export OPENAI_API_KEY=pplx-*************
```

> **Note:** Replace `pplx-*************` with your actual Perplexity API key. Keep your API key confidential and do not share it publicly.

### Step 3: Install required modules

Install all necessary modules in your virtual environment:

```bash
pip install -r requirements.txt
```

Ensure you have a `requirements.txt` file in your project directory with all the required dependencies listed.

### Step 4: Run the code

Execute the main Python script:

```bash
python FinanceAnalystPerplexityBasedAgenticWorkflow.py
```

---

## Additional Information

- Ensure you have Python 3.x installed on your system.
- For any issues or questions, please open an issue in this repository.

---

## Happy analyzing!
