# Multi-Domain Support Triage Agent

## Overview

This agent processes support tickets for HackerRank, Claude, and Visa using only the provided support corpus to generate grounded responses. It implements a RAG (Retrieval-Augmented Generation) approach with keyword-based retrieval.

## Architecture

### Components

1. **SupportCorpusIndex** (`agent.py`)
   - Loads all markdown documents from the `data/` directory
   - Parses document metadata (title, breadcrumbs, source URL)
   - Implements keyword-based search with TF-IDF-like scoring
   - Supports company-specific filtering

2. **SupportTriageAgent** (`agent.py`)
   - Detects company from ticket content or explicit field
   - Classifies product area using keyword matching
   - Determines escalation vs. reply based on risk assessment
   - Generates grounded responses from retrieved documents

### Escalation Criteria

The agent escalates tickets containing:
- Security vulnerabilities / bug bounty reports
- Account access issues requiring admin intervention
- Fraud/identity theft claims
- Disputes requiring human review (score disputes, merchant bans)
- Requests outside scope of support corpus
- Legal/compliance matters
- System-wide outages
- Requests for internal information

### Product Areas

**HackerRank:** screen, interview, community, billing, integrations, engage, chakra, skillup, candidates, settings

**Claude:** account_management, conversation_management, billing, api_console, privacy, features, mobile, desktop, code, security

**Visa:** card_services, travel_support, fraud_dispute, merchant_support, general_support

## Usage

```bash
# Run the agent
python3 code/agent.py
```

### Input
- `support_tickets/support_tickets.csv` - Contains support tickets with columns: Issue, Subject, Company

### Output
- `support_tickets/output.csv` - Generated responses with columns:
  - `issue`: Original issue text
  - `subject`: Ticket subject
  - `company`: Detected/inferred company
  - `response`: User-facing response grounded in corpus
  - `product_area`: Classified support category
  - `status`: `replied` or `escalated`
  - `request_type`: `product_issue`, `feature_request`, `bug`, or `invalid`
  - `justification`: Explanation of decision

## Requirements

- Python 3.8+
- No external dependencies (uses only standard library)

## Environment Variables

No environment variables required. The agent uses only the local support corpus.

## Determinism

The agent is fully deterministic:
- No random sampling
- Seeded where applicable
- Consistent output for same input

## Engineering Hygiene

- Secrets read from environment variables only (none required for this implementation)
- No hardcoded API keys
- Readable, modular code structure
- Type hints throughout
- Comprehensive docstrings

## Files

- `code/agent.py` - Main agent implementation
- `code/main.py` - Entry point (calls agent.py main function)
- `data/` - Support corpus (774 documents across HackerRank, Claude, Visa)
- `support_tickets/support_tickets.csv` - Input tickets
- `support_tickets/output.csv` - Generated output
