# Finance Sentiment Chain — README

A minimal LangChain + Azure OpenAI pipeline that:

1. takes a company name as input, 2) resolves the public stock ticker, 3) fetches recent news via Yahoo Finance (`yfinance`), 4) asks an LLM to produce a structured JSON sentiment report, and 5) logs prompts and outputs to MLflow.

---

## 1) Prerequisites

* **Python** 3.10+
* **Azure OpenAI** resource with a deployed chat model (e.g., a deployment named `gpt4o`)
* **MLflow Tracking Server** (or local tracking). The sample code points to `http://20.75.92.162:5000` — ensure you can reach this endpoint
* **Internet access** from your machine to reach Azure OpenAI and Yahoo Finance

---

## 2) Quick Start

### 2.1. Clone & create a virtual environment

```bash
# clone your repo (example)
git clone <your-repo-url> finance-sentiment-chain
cd finance-sentiment-chain

# create and activate venv (Linux/macOS)
python -m venv .venv
source .venv/bin/activate

# on Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2.2. Install dependencies

Create a `requirements.txt` with the following (or use your own lockfile):

```txt
langchain
langchain-core
langchain-openai
mlflow
yfinance
pydantic>=2
python-dotenv  # optional, for local env var loading
```

Then install:

```bash
pip install -r requirements.txt
```

> **Note:** `yfinance` pulls news via Yahoo Finance and may also install `pandas`, `numpy`, etc.

---

## 3) Azure OpenAI Configuration

The provided code uses:

```python
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(model="gpt4o")
```

With **LangChain + Azure OpenAI**, the common configuration patterns are:

### Option A — Environment variables (recommended)

Set these env vars before running the script:

```bash
# core
export AZURE_OPENAI_API_KEY="<your-azure-openai-key>"
export AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-05-01-preview"  # or the version for your deployment

# deployment name mapping used by langchain_openai
export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt4o"  # must match your Azure deployment name
```

On Windows (PowerShell):

```powershell
setx AZURE_OPENAI_API_KEY "<your-azure-openai-key>"
setx AZURE_OPENAI_ENDPOINT "https://<your-resource-name>.openai.azure.com"
setx AZURE_OPENAI_API_VERSION "2024-05-01-preview"
setx AZURE_OPENAI_CHAT_DEPLOYMENT "gpt4o"
```

> If your actual Azure **deployment name** is different (e.g., `gpt-4o-chat`), set `AZURE_OPENAI_CHAT_DEPLOYMENT` to that value. The code passes `model="gpt4o"`; LangChain resolves to the env deployment when `AZURE_OPENAI_CHAT_DEPLOYMENT` is set.

### Option B — Pass the deployment name in code

Change the constructor to explicitly target your deployment:

```python
llm = AzureChatOpenAI(azure_deployment="gpt4o", api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"))
```

Keep your `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` as env vars.

---

## 4) MLflow Configuration

The script configures MLflow as follows:

```python
mlflow.set_tracking_uri("http://20.75.92.162:5000")
mlflow.set_experiment("Finance_Sentiment_Anlysis_RohitZ")
```

Make sure:

* The tracking server URL is reachable and correct.
* Your MLflow server allows client logging from your network.
* If authentication is enabled, set the required env vars (`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`) or a token as needed.

### Optional environment variables

```bash
export MLFLOW_TRACKING_URI="http://20.75.92.162:5000"
export MLFLOW_EXPERIMENT_NAME="Finance_Sentiment_Anlysis_RohitZ"
```

If you set these, you can omit the hard-coded calls in code and read from `os.environ` instead.

### Viewing runs

Open the MLflow UI in your browser:

```
http://20.75.92.162:5000
```

You should see experiment **Finance\_Sentiment\_Anlysis\_RohitZ** and individual runs. Each run logs:

* `company_name` as a parameter
* prompt templates (saved under `prompts/` as JSON artifacts)
* the final output JSON at `artifacts/final_output.json`

---

## 5) Yahoo Finance (yfinance) Notes

* The chain calls `yf.Ticker(<ticker>).news` to fetch recent news items.
* If no news is available, the chain inserts a placeholder record (`"No recent articles found."`).
* Occasionally, Yahoo’s response can be empty or rate-limited; rerun with a different company or later if needed.

---

## 6) Run the Chain

Assuming you saved the provided script as `finance_chain.py`:

```bash
python finance_chain.py
```

You will be prompted:

```
Enter company name :
```

Provide a name (e.g., `Microsoft`), then the chain will:

1. Resolve the ticker/exchange via the LLM
2. Fetch news via `yfinance`
3. Ask the LLM for a structured sentiment report
4. Log parameters, prompts, and final JSON to MLflow

**Sample** (illustrative):

```json
{
  "company": "Microsoft",
  "stock_code": "MSFT",
  "ticker_info": {"company": "Microsoft", "ticker": "MSFT", "exchange": "NASDAQ", "confidence": "high"},
  "analysis": {
    "company_name": "Microsoft",
    "stock_code": "MSFT",
    "newsdesc": "...",
    "sentiment": "Neutral",
    "people_names": ["..."] ,
    "places_names": ["..."] ,
    "other_companies_referred": ["..."],
    "related_industries": ["..."],
    "market_implications": "...",
    "confidence_score": 0.7
  }
}
```

---

## 7) Suggested `.env` (optional)

Create a `.env` file to avoid exporting variables every time:

```env
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt4o
MLFLOW_TRACKING_URI=http://20.75.92.162:5000
MLFLOW_EXPERIMENT_NAME=Finance_Sentiment_Anlysis_RohitZ
```

Then load it at the start of your script:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 8) Troubleshooting

* **Auth errors (401/403) with Azure OpenAI**: Verify key, endpoint, API version, and that your deployment name matches `AZURE_OPENAI_CHAT_DEPLOYMENT`.
* **Model not found**: Ensure the Azure deployment exists and is a chat-capable model compatible with `AzureChatOpenAI`.
* **MLflow not reachable**: Confirm network/firewall allows access to `http://20.75.92.162:5000` and that the MLflow server is running.
* **`yfinance` returns no news**: Try another well-covered ticker, or rerun later.
* **JSON parsing errors**: The chain attempts to coerce model output to valid JSON. If the model drifts, consider adding more explicit formatting instructions or using a JSON output parser with strict schemas.

---

## 9) Security & Compliance

* Do not commit secrets to source control. Use environment variables or secret managers.
* If your MLflow server is public, protect it with auth and TLS.
* Review Azure OpenAI data handling policies and regional requirements for your tenant.

---

## 10) Repository Structure (suggested)

```
.
├─ finance_chain.py                 # the script in this README
├─ requirements.txt
├─ README.md                        # this file
├─ .env.example                     # sample env file (no secrets)
└─ prompts/                         # prompt templates logged to MLflow
```

---

## 11) Sample One-Liners

* **Unix/macOS** (with inline env vars):

```bash
AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_VERSION=2024-05-01-preview \
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt4o MLFLOW_TRACKING_URI=http://20.75.92.162:5000 \
python finance_chain.py
```

* **Windows PowerShell** (after `setx` once; reopen shell):

```powershell
python .\finance_chain.py
```

---

**You’re set!** Run the script, type a company name, and review the run in MLflow.

