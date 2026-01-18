# Shadow-Optic Scripts

Utility scripts for bootstrapping, testing, and demo preparation.

##  Scripts Overview

### 1. `bootstrap_portkey.py` - Configuration Setup
**Purpose:** Upload Shadow-Optic configurations to Portkey and capture config IDs.

**When to run:** Before starting Shadow-Optic for the first time.

**Usage:**
```bash
# With environment variable
export PORTKEY_API_KEY=your_key
python scripts/bootstrap_portkey.py

# Or with flag
python scripts/bootstrap_portkey.py --api-key YOUR_KEY
```

**What it does:**
- Uploads `configs/production-config.json` to Portkey
- Uploads `configs/shadow-config.json` to Portkey
- Verifies configs exist in Portkey
- Writes config IDs to `.env` file

**Output:**
- `PORTKEY_PRODUCTION_CONFIG_ID` in `.env`
- `PORTKEY_SHADOW_CONFIG_ID` in `.env`

---

### 2. `seed_traffic.py` - Demo Data Generator
**Purpose:** Generate diverse production traffic for realistic shadow testing.

**When to run:** After starting Shadow-Optic, before triggering optimization.

**Usage:**
```bash
# Generate 50 diverse requests (default)
python scripts/seed_traffic.py

# Custom count
python scripts/seed_traffic.py --count 100 --concurrency 10

# Specify model
python scripts/seed_traffic.py --model gpt-5.2 --count 50
```

**What it does:**
- Sends diverse prompts across 7 categories:
  - Coding (Python, JS, SQL)
  - Writing (emails, blog posts)
  - Analysis (comparisons, trade-offs)
  - Math & Logic
  - Business (metrics, strategy)
  - Conceptual (explanations)
  - Debugging (troubleshooting)
- Routes through Portkey production config
- Displays progress and statistics

**Parameters:**
- `--count`: Number of requests (default: 50)
- `--concurrency`: Parallel requests (default: 5)
- `--model`: Model to use (default: gpt-5.2)
- `--api-key`: Portkey API key
- `--config-id`: Production config ID

---

### 3. `preflight_check.py` - Comprehensive Verification
**Purpose:** Verify all components are ready before demo.

**When to run:** Before any demo or production deployment.

**Usage:**
```bash
python scripts/preflight_check.py
```

**What it checks:**
-  Environment variables set
-  Python packages installed
-  Portkey API connectivity
-  Qdrant connectivity
-  Temporal connectivity
-  Portkey configs exist

**Output:** Beautiful terminal report with pass/fail status for each component.

---

##  Quick Start Workflow

### First Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Bootstrap Portkey configs
export PORTKEY_API_KEY=your_key_here
python scripts/bootstrap_portkey.py

# 3. Start services
docker-compose up -d

# 4. Verify everything is ready
python scripts/preflight_check.py

# 5. Generate seed traffic
python scripts/seed_traffic.py --count 50

# 6. Trigger optimization
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "challenger_models": ["deepseek-chat", "gemini-3-flash", "haiku-4.5"]
  }'

# 7. Monitor workflow
open http://localhost:8233  # Temporal UI
open http://localhost:3000  # Grafana
```

---

##  Demo Preparation Checklist

- [ ] **Environment Variables**
  - `PORTKEY_API_KEY`
  - `PORTKEY_PRODUCTION_CONFIG_ID`
  - `PORTKEY_SHADOW_CONFIG_ID`
  - `QDRANT_URL` (optional, defaults to localhost)

- [ ] **Services Running**
  - Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
  - Temporal: `temporal server start-dev`
  - Shadow-Optic API: `docker-compose up`

- [ ] **Data Generation**
  - Seed traffic: 50+ diverse requests
  - Wait 2-3 minutes for logs to sync

- [ ] **Verification**
  - Run preflight check
  - Check Portkey dashboard for logs
  - Verify Qdrant has collections

---

##  Hackathon "Money Shot" Sequence

For maximum impact during demo presentation:

```bash
# 1. Show clean slate
open https://app.portkey.ai/logs  # Should be empty or old data

# 2. Generate diverse traffic (live!)
python scripts/seed_traffic.py --count 30

# 3. Show logs appearing in Portkey
open https://app.portkey.ai/logs  # Now populated!

# 4. Trigger optimization workflow
curl -X POST http://localhost:8000/api/v1/optimize

# 5. Show Temporal workflow executing
open http://localhost:8233  # Watch activities progress

# 6. Wait for completion (~2-5 minutes)
# Activities shown:
#   1. Export Portkey logs 
#   2. Semantic sampling 
#   3. Shadow replays 
#   4. LLM-as-Judge evaluation 
#   5. Cost analysis 
#   6. Recommendation generation 

# 7. Show results in Portkey Feedback
open https://app.portkey.ai/feedback  # Shadow scores!

# 8. Show Grafana dashboards
open http://localhost:3000  # Cost savings metrics
```

---

##  Troubleshooting

### Bootstrap fails with "Config not found"
**Solution:** Make sure your Portkey API key has permissions to create configs.

### Seed traffic fails with rate limiting
**Solution:** Reduce `--concurrency` or add delays between requests.

### Preflight check shows Temporal not reachable
**Solution:** 
```bash
# Start Temporal locally
temporal server start-dev

# Or check if running
ps aux | grep temporal
```

### No logs appearing in Portkey after traffic
**Solution:** Wait 2-3 minutes. Portkey logs have a slight delay for batching.

---

##  Notes

- **Costs:** Seed traffic uses real API calls. 50 requests ≈ $0.50-$1.00 depending on model.
- **Rate Limits:** Default concurrency (5) is conservative. Increase carefully.
- **Portkey Delay:** Logs take 1-3 minutes to appear in dashboard.
- **Demo Timing:** Full workflow takes 2-5 minutes depending on sample size.

---

## 🆘 Support

If scripts fail:
1. Check `preflight_check.py` output
2. Verify `.env` file has all required variables
3. Check docker containers are running: `docker ps`
4. Check Temporal UI: http://localhost:8233
5. Check application logs: `docker-compose logs -f shadow-optic-api`
