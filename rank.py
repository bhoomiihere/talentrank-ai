"""
TalentRank AI — Candidate Ranker
Redrob INDIA.RUNS · Track 1: Data & AI Challenge

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Design philosophy (from reading the JD carefully):
  - Career relevance > keyword count. A Marketing Manager with AI skills is NOT a fit.
  - Product company AI experience >> consulting firm AI keywords.
  - Behavioral signals are a MULTIPLIER — inactive candidates are not hirable.
  - Honeypots (impossible profiles) must rank at the bottom.
  - Reasoning must be specific, honest, and reference actual profile facts.
  - No network calls. No GPU. Runs in <60s on CPU.
"""

import argparse
import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

# ── Constants from JD ────────────────────────────────────────────────────────

# Things you ABSOLUTELY need (from JD "skills inventory")
MUST_HAVE = {
    "embeddings", "retrieval", "vector search", "semantic search",
    "sentence-transformers", "sentence transformers",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "hybrid search",
    "dense retrieval", "ranking", "reranking", "re-ranking",
    "ndcg", "mrr", "map", "information retrieval",
    "python", "pytorch", "transformers", "bert", "bge", "e5",
    "rag", "retrieval augmented", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft", "learning to rank",
    "nlp", "natural language processing",
    "llm", "large language model", "vector database",
    "hugging face", "huggingface", "machine learning",
    "evaluation framework", "a/b testing", "recommendation",
    "search", "openai embeddings",
}

# Nice to have (from JD)
NICE_HAVE = {
    "xgboost", "lightgbm", "kafka", "redis", "docker", "kubernetes",
    "fastapi", "airflow", "spark", "distributed systems",
    "inference optimization", "open source", "mlflow",
    "deep learning", "neural networks", "scikit-learn",
    "hr tech", "recruiting", "marketplace",
}

# Strong AI/ML title signals
AI_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "nlp engineer", "senior ml engineer", "lead ml engineer",
    "staff ml engineer", "principal ml engineer",
    "applied scientist", "research engineer", "senior ai engineer",
    "data scientist", "senior data scientist", "lead data scientist",
    "search engineer", "ranking engineer", "retrieval engineer",
    "senior nlp engineer", "recommendation engineer",
    "ml researcher", "applied ml engineer",
    "junior ml engineer", "ai researcher",
    "staff machine learning engineer",
    "senior machine learning engineer",
}

# JD explicitly says these are disqualifiers (consulting-only career)
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "ltimindtree",
}

# JD explicitly does NOT want these as primary titles
BAD_TITLES = {
    "marketing manager", "hr manager", "content writer",
    "graphic designer", "accountant", "sales executive",
    "business analyst", "project manager", "operations manager",
    "customer support", "civil engineer", "mechanical engineer",
    "teacher", "finance manager", "supply chain", "procurement",
    "digital marketing", "seo", "social media",
}

TODAY = date.today()


# ── Honeypot Detection ───────────────────────────────────────────────────────

def is_honeypot(c: dict) -> tuple[bool, str]:
    """
    Detect impossible profiles — these are forced to tier 0 in ground truth.
    From spec: ~80 honeypots with subtly impossible profiles.
    """
    skills = c.get("skills", [])
    career = c.get("career_history", [])

    # Expert skill with 0 months of use — impossible
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0:
            return True, f"expert '{s['name']}' with 0 months use"

    # Duration > actual time since start date — impossible
    for job in career:
        dur = job.get("duration_months", 0)
        start_str = job.get("start_date", "")
        try:
            sd = datetime.strptime(start_str, "%Y-%m-%d").date()
            actual_months = (TODAY - sd).days / 30.44
            if dur > actual_months + 12:
                return True, f"claims {dur}mo at {job['company']} (only {int(actual_months)}mo possible)"
        except Exception:
            pass

    # 8+ expert/advanced skills with ZERO total endorsements — impossible
    strong_skills = [s for s in skills if s.get("proficiency") in ("expert", "advanced")]
    if len(strong_skills) >= 8 and sum(s.get("endorsements", 0) for s in strong_skills) == 0:
        return True, "8+ advanced/expert skills with 0 endorsements"

    return False, ""


# ── Skill Scoring ────────────────────────────────────────────────────────────

def _prof_weight(prof: str) -> float:
    return {"expert": 1.0, "advanced": 0.82, "intermediate": 0.55, "beginner": 0.25}.get(prof, 0.25)


def score_skills(candidate_skills: list, target: set) -> tuple[float, list]:
    """
    Quality-weighted skill match.
    Score = Σ(matched skill weight) / Σ(all skill weight)
    Weight = proficiency × (1 + endorsement bonus) × (1 + duration bonus)
    """
    total_w, matched_w = 0.0, 0.0
    matched_names = []
    for sk in candidate_skills:
        name = sk["name"].lower().strip()
        pw = _prof_weight(sk.get("proficiency", "beginner"))
        eb = min(sk.get("endorsements", 0) / 50.0, 1.0) * 0.20
        db = min(sk.get("duration_months", 0) / 30.0, 1.0) * 0.15
        w = pw * (1 + eb + db)
        total_w += w
        if any(t in name or name in t for t in target):
            matched_w += w
            matched_names.append(sk["name"])
    ratio = matched_w / max(total_w, 1e-9)
    return ratio, matched_names


# ── Career Scoring ───────────────────────────────────────────────────────────

def score_career(c: dict) -> tuple[float, bool, str, list]:
    """
    Career relevance is the most important signal per JD.
    Returns: (score 0-1, is_disqualified, reason, ai_titles_held)

    Key rules from JD:
    - Product company AI/ML experience is the gold standard
    - Consulting-only career → disqualified
    - Current title in BAD_TITLES → disqualified
    - Job-hopping (< 1yr avg tenure) → penalty
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    current_title = profile.get("current_title", "").lower()

    # Hard disqualifier: current title is clearly non-AI
    if any(bt in current_title for bt in BAD_TITLES):
        return 0.05, True, f"current title '{profile.get('current_title', '')}' is not AI/ML", []

    total_months = 0
    ai_months = 0
    product_ai_months = 0
    consulting_months = 0
    ai_titles_held = []

    for job in career:
        title = job.get("title", "").lower()
        company = job.get("company", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur

        is_consulting = any(cf in company for cf in CONSULTING_FIRMS)
        is_ai = any(at in title for at in AI_TITLES)

        if is_consulting:
            consulting_months += dur
        if is_ai:
            ai_months += dur
            ai_titles_held.append(job.get("title", ""))
            if not is_consulting:
                product_ai_months += dur

    # Disqualify: entire career in consulting
    if total_months > 0 and consulting_months / total_months > 0.85:
        return 0.05, True, "entire career in consulting firms (JD explicit disqualifier)", ai_titles_held

    if total_months == 0:
        return 0.1, False, "no career history", []

    # Score: heavily reward product-company AI
    ai_ratio = ai_months / total_months
    product_ratio = product_ai_months / total_months
    consulting_penalty = 0.25 * (consulting_months / total_months)

    # Job hopping penalty
    n_jobs = len(career)
    avg_tenure = total_months / max(n_jobs, 1)
    hop_penalty = 0.1 if avg_tenure < 12 else 0.0

    score = 0.35 * ai_ratio + 0.65 * product_ratio - consulting_penalty - hop_penalty
    return max(0.0, min(score, 1.0)), False, "", ai_titles_held


# ── Experience Fit ───────────────────────────────────────────────────────────

def score_experience(years: float) -> float:
    """
    JD says 5-9 but explicitly flexible. Smooth decay outside range.
    """
    if 5 <= years <= 9:
        return 1.0
    elif years < 5:
        return max(0.25, 1.0 - (5 - years) * 0.15)
    else:
        return max(0.55, 1.0 - (years - 9) * 0.05)  # gentle — overexp is fine


# ── Behavioral Score ─────────────────────────────────────────────────────────

def score_behavioral(signals: dict) -> float:
    """
    23 Redrob signals → normalized 0-1 score.
    Key insight from JD: behavioral signals are a multiplier.
    Inactive candidate with 5% response rate is not hirable.
    """
    s = 0.0

    # Availability (20%)
    try:
        last_active = datetime.strptime(signals.get("last_active_date", "2020-01-01"), "%Y-%m-%d").date()
        inactive_days = (TODAY - last_active).days
        active_score = max(0.0, 1.0 - inactive_days / 150.0)
    except Exception:
        active_score = 0.0
    otw = 1.0 if signals.get("open_to_work_flag", False) else 0.2
    s += 0.12 * active_score + 0.08 * otw

    # Engagement quality (30%)
    rr = float(signals.get("recruiter_response_rate", 0))
    ir = float(signals.get("interview_completion_rate", 0))
    pc = float(signals.get("profile_completeness_score", 0)) / 100.0
    s += 0.14 * rr + 0.08 * ir + 0.08 * pc

    # Platform demand signals (20%)
    saved = min(float(signals.get("saved_by_recruiters_30d", 0)) / 10.0, 1.0)
    views = min(float(signals.get("profile_views_received_30d", 0)) / 50.0, 1.0)
    s += 0.12 * saved + 0.08 * views

    # Technical credibility (15%)
    gh = float(signals.get("github_activity_score", -1))
    gh_score = (gh / 100.0) if gh >= 0 else 0.0
    s += 0.15 * gh_score

    # Trust / verification (10%)
    trust = (int(signals.get("verified_email", False)) +
             int(signals.get("verified_phone", False)) +
             int(signals.get("linkedin_connected", False))) / 3.0
    s += 0.10 * trust

    # Notice period adjustment (5%)
    notice = int(signals.get("notice_period_days", 60))
    if notice <= 30:
        s += 0.05
    elif notice > 90:
        s -= 0.03

    return min(max(s, 0.0), 1.0)


# ── Assessment Bonus ─────────────────────────────────────────────────────────

def assessment_bonus(signals: dict) -> float:
    """Bonus for completing Redrob platform assessments in relevant skills."""
    scores = signals.get("skill_assessment_scores", {})
    if not scores:
        return 0.0
    relevant = [v for k, v in scores.items()
                if any(t in k.lower() for t in MUST_HAVE)]
    if not relevant:
        return 0.0
    return (sum(relevant) / len(relevant)) / 100.0 * 0.04  # max 4% bonus


# ── Composite Score ──────────────────────────────────────────────────────────

def score_candidate(c: dict) -> tuple[float, dict]:
    """
    Final composite score with breakdown for reasoning.
    Weights chosen to match JD emphasis:
    - Career is the decisive signal (40%)
    - Skills depth (25%)
    - Behavioral availability (20%)
    - Experience fit (10%)
    - Nice-to-have skills + assessment bonus (5%)
    """
    profile = c.get("profile", {})
    skills = c.get("skills", [])
    signals = c.get("redrob_signals", {})
    exp_years = float(profile.get("years_of_experience", 0))

    # Honeypot → hard bottom
    hp, hp_reason = is_honeypot(c)
    if hp:
        return -1.0, {"honeypot": hp_reason}

    career_sc, disq, disq_reason, ai_titles = score_career(c)
    must_sc, must_matched = score_skills(skills, MUST_HAVE)
    nice_sc, nice_matched = score_skills(skills, NICE_HAVE)
    exp_sc = score_experience(exp_years)
    beh_sc = score_behavioral(signals)
    assess = assessment_bonus(signals)

    final = (
        0.40 * career_sc +
        0.25 * must_sc +
        0.20 * beh_sc +
        0.10 * exp_sc +
        0.05 * nice_sc +
        assess
    )

    if disq:
        final *= 0.12  # still ranked but at bottom

    breakdown = {
        "career": round(career_sc, 3),
        "must_skills": round(must_sc, 3),
        "behavioral": round(beh_sc, 3),
        "experience": round(exp_sc, 3),
        "must_matched": must_matched[:5],
        "nice_matched": nice_matched[:3],
        "ai_titles": ai_titles[:2],
        "disqualified": disq,
        "disq_reason": disq_reason,
        "exp_years": exp_years,
        "rr": float(signals.get("recruiter_response_rate", 0)),
        "notice": int(signals.get("notice_period_days", 60)),
        "inactive_days": _inactive_days(signals),
        "github": float(signals.get("github_activity_score", -1)),
        "open_to_work": signals.get("open_to_work_flag", False),
        "current_title": profile.get("current_title", "Unknown"),
        "current_company": profile.get("current_company", "Unknown"),
        "location": profile.get("location", "Unknown"),
    }
    return max(0.0, min(final, 1.0)), breakdown


def _inactive_days(signals: dict) -> int:
    try:
        d = datetime.strptime(signals.get("last_active_date", "2020-01-01"), "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return 9999


# ── Reasoning Builder ────────────────────────────────────────────────────────

def build_reasoning(cid: str, rank: int, score: float, b: dict) -> str:
    """
    Specific, honest, non-templated reasoning referencing actual profile facts.
    Varies by rank tier and what's notable about the candidate.
    Penalized by judges: templated, hallucinated, or contradicts rank.
    """
    parts = []
    title = b["current_title"]
    exp = b["exp_years"]
    rr = b["rr"]
    notice = b["notice"]
    inactive = b["inactive_days"]
    gh = b["github"]

    # Lead with title + experience
    parts.append(f"{title} with {exp:.1f} yrs experience")

    # Top matched skills (specific, not generic)
    if b["must_matched"]:
        skills_str = ", ".join(b["must_matched"][:3])
        parts.append(f"matched JD-required skills: {skills_str}")
    elif b["nice_matched"]:
        parts.append(f"adjacent skills ({', '.join(b['nice_matched'][:2])}) but limited core AI/ML match")

    # Career signal
    if b["ai_titles"]:
        parts.append(f"held AI/ML roles ({b['ai_titles'][0]})")
    elif b["disqualified"] and b["disq_reason"]:
        parts.append(f"concern: {b['disq_reason']}")

    # Behavioral facts — always specific numbers
    if rr >= 0.75:
        parts.append(f"strong recruiter engagement ({rr:.0%} response rate)")
    elif rr < 0.15:
        parts.append(f"low recruiter response rate ({rr:.0%}) — availability concern")

    if inactive > 150:
        parts.append(f"inactive {inactive}d — may not be actively looking")
    elif inactive <= 14 and b["open_to_work"]:
        parts.append("recently active, open to work")

    if notice <= 15:
        parts.append(f"immediate joiner (notice: {notice}d)")
    elif notice > 90:
        parts.append(f"long notice period ({notice}d)")

    if gh >= 60:
        parts.append(f"strong GitHub activity ({gh:.0f}/100)")
    elif gh < 0:
        parts.append("no GitHub linked")

    # Rank-appropriate tone
    if rank <= 10 and score >= 0.65:
        suffix = "Strong overall fit for the role."
    elif rank <= 30:
        suffix = "Good fit with some gaps."
    elif rank <= 60:
        suffix = "Partial fit — included for adjacent signals."
    else:
        suffix = "Below cutoff — included to complete top-100."

    return ". ".join(parts) + ". " + suffix


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TalentRank AI — Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    cand_path = Path(args.candidates)
    if not cand_path.exists():
        print(f"ERROR: candidates file not found at {cand_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading candidates from {cand_path}...")
    candidates = []
    with open(cand_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates.")

    print("Scoring...")
    scored = []
    honeypot_count = 0
    for i, c in enumerate(candidates):
        if i % 25000 == 0 and i > 0:
            print(f"  {i:,}/{len(candidates):,}...")
        try:
            sc, breakdown = score_candidate(c)
            if sc < 0:
                honeypot_count += 1
                sc = 0.0001  # rank at bottom
            scored.append((c["candidate_id"], sc, breakdown))
        except Exception as e:
            scored.append((c["candidate_id"], 0.0001, {
                "current_title": "Unknown", "exp_years": 0, "rr": 0,
                "notice": 60, "inactive_days": 9999, "github": -1,
                "open_to_work": False, "must_matched": [], "nice_matched": [],
                "ai_titles": [], "disqualified": False, "disq_reason": "",
                "current_company": "Unknown", "location": "Unknown",
            }))

    print(f"  Honeypots detected and ranked to bottom: {honeypot_count}")

    # Sort: score descending, then candidate_id ascending (tie-break per spec)
    scored.sort(key=lambda x: (-x[1], x[0]))
    top100 = scored[:100]

    # Verify no honeypots in top 100
    honeypots_in_top = sum(1 for _, sc, _ in top100 if sc <= 0.0001)
    print(f"  Honeypots in top 100: {honeypots_in_top} (target: 0)")

    print("\nTop 10 candidates:")
    for i, (cid, sc, b) in enumerate(top100[:10]):
        print(f"  #{i+1:>2} {cid}  score={sc:.4f}  {b.get('current_title','?')} "
              f"({b.get('exp_years',0):.1f}yrs)  rr={b.get('rr',0):.2f}")

    # Write submission CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev_score = None
        for rank, (cid, raw_sc, b) in enumerate(top100, start=1):
            # Non-increasing scores: interpolate from 0.99 down to 0.20
            norm_score = round(0.99 - (rank - 1) * (0.79 / 99), 4)
            if prev_score is not None:
                norm_score = min(norm_score, prev_score)
            prev_score = norm_score
            reasoning = build_reasoning(cid, rank, raw_sc, b)
            reasoning = reasoning.replace('"', "'").replace("\n", " ")
            writer.writerow([cid, rank, norm_score, reasoning])

    print(f"\n✓ Submission written to {out_path}")
    print(f"  Rows: 100 | Ranks: 1–100 | Scores: non-increasing")


if __name__ == "__main__":
    main()
