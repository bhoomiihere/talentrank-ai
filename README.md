# TalentRank AI
### Redrob INDIA.RUNS — Track 1: Data & AI Challenge

---

## What This Does

Ranks 100,000 candidates for the Redrob Senior AI Engineer role the way a great recruiter would — not by counting keywords, but by understanding who actually fits.

**Single command to reproduce:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runtime: ~50 seconds · CPU only · No GPU · No network calls · No external dependencies

---

## The Core Insight

The JD explicitly warns against keyword matching:

> *"The right answer involves reasoning about the gap between what the JD says and what the JD means. A candidate who has all the AI keywords listed as skills but whose title is 'Marketing Manager' is not a fit."*

So the ranker is built around three principles:

**1. Career > Keywords**  
What title did they hold? At what kind of company? For how long? A Search Engineer at a product startup with 4 years of retrieval experience outranks a Marketing Manager with 10 AI keywords in their skills list.

**2. Behavioral signals = availability**  
A perfect-on-paper candidate who hasn't logged in for 6 months with a 5% recruiter response rate is not hirable. Behavioral signals multiply the score — they don't just add to it.

**3. Honeypots rank at the bottom**  
Profiles with impossible facts (expert skill with 0 months use, duration > company age) are detected and ranked to the bottom. Zero honeypots in our top 100.

---

## Scoring Pipeline

```
candidates.jsonl (100K)
        ↓
  Honeypot Detection       is_honeypot() → ranks to bottom
        ↓
  Career Relevance (40%)   AI/ML titles at product companies
        ↓
  Skill Quality    (25%)   proficiency × endorsements × duration
        ↓
  Behavioral Score (20%)   7 Redrob signals: response rate, activity, GitHub
        ↓
  Experience Fit   (10%)   smooth decay 5-9yr range
        ↓
  Nice-to-have     (5%)    secondary skill signals
        ↓
  submission.csv           top 100, validated ✓
```

---

## Scoring Weights

| Component | Weight | Rationale |
|---|---|---|
| Career relevance | 40% | JD's decisive signal — AI/ML titles at product cos |
| Skill quality | 25% | Depth-weighted (proficiency × endorsements × duration) |
| Behavioral signals | 20% | Availability, engagement, GitHub, verification |
| Experience fit | 10% | Smooth decay, flexible per JD guidance |
| Nice-to-have skills | 5% | Secondary signal |

---

## Key Design Decisions

**Why career scores 40%?**  
The JD spends more words on career trajectory than on skills. It explicitly disqualifies: pure consulting backgrounds (TCS/Infosys/Wipro/Accenture), pure research roles, and non-AI current titles. Career is the filter; skills are the validator.

**Why depth-weighted skills?**  
`skill_weight = proficiency_weight × (1 + endorsements/50 × 0.2) × (1 + duration_months/30 × 0.15)`  
Expert + 45 endorsements + 36 months >> Beginner + 0 endorsements + 0 months. Keyword presence alone scores very low.

**Why behavioral signals matter so much?**  
From the signals doc: *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available."* We treat this seriously — behavioral score multiplies availability into the final rank.

**Why no ML model?**  
CPU-only, no network, 5-minute budget, 100K candidates. A well-designed scoring function is interpretable, reproducible, and takes 50 seconds. It also survives the defend-your-work interview because every line of logic maps directly to a JD requirement.

---

## Validation

```
✓ Submission is valid.          (validate_submission.py)
✓ 100 rows                      (exactly 100 candidates)
✓ Ranks 1–100                   (each appears exactly once)
✓ Scores non-increasing         (0.99 → 0.20)
✓ All CAND_XXXXXXX IDs valid    (from candidates.jsonl)
✓ Honeypots in top 100: 0       (40 detected, all at bottom)
✓ Runtime: ~50 seconds          (well within 5-min limit)
✓ Zero external dependencies    (pure Python stdlib)
```

---

## Top 10 Ranked Candidates

| Rank | Candidate | Score | Title | Why |
|---|---|---|---|---|
| 1 | CAND_0042100 | 0.990 | ML Engineer (7.3yr) | Elasticsearch, LTR, Recommendation; 87% response |
| 2 | CAND_0018499 | 0.982 | Senior ML Engineer (7.2yr) | Weaviate, Pinecone; 15d notice; GitHub 95/100 |
| 3 | CAND_0071974 | 0.974 | Senior AI Engineer (7.8yr) | LoRA, LTR, Weaviate; 76% response; GitHub 83 |
| 4 | CAND_0036437 | 0.966 | Search Engineer (4.8yr) | OpenSearch, Elasticsearch, ST; 87% response |
| 5 | CAND_0086022 | 0.958 | Senior Applied Scientist (5.3yr) | Vector Search, Fine-tuning; immediate joiner |

---

## Setup & Run

```bash
# No dependencies — pure Python stdlib
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate output
python validate_submission.py submission.csv
# → Submission is valid.
```

**Requirements:** Python 3.8+ · CPU only · ~2GB RAM · No network · No GPU

---

## Project Structure

```
talentrank-ai/
├── rank.py                    ← Single ranker file — run this
├── submission.csv             ← Validated ranked output
├── submission_metadata.yaml   ← Team metadata
├── requirements.txt           ← No external deps (pure stdlib)
└── README.md
```

---

## AI Tools Declaration

Claude (Anthropic) was used for architecture discussion and code review.
All scoring logic, signal weighting, JD analysis, and pipeline design are original engineering work.
No candidate data was processed by any external LLM.
Ranking runs fully offline — zero API calls, zero network during inference.

---

*Built for Redrob INDIA.RUNS · Track 1: Data & AI Challenge*
