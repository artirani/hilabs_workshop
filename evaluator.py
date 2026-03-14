"""
evaluator.py — Clinical entity evaluation logic.

Combines:
  1. Rule-based heuristics (fast, deterministic)
  2. LLM-based scoring via OpenRouter API (deep reasoning)

Set your key before running:
    export OPENROUTER_API_KEY=sk-or-v1-...
"""

import json
import re
import os
import time
import requests
from collections import defaultdict
from typing import Any


# ── Constants ────────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = {
    "MEDICINE", "PROBLEM", "PROCEDURE", "TEST",
    "VITAL_NAME", "IMMUNIZATION", "MEDICAL_DEVICE",
    "MENTAL_STATUS", "SDOH", "SOCIAL_HISTORY",
}

VALID_ASSERTIONS   = {"POSITIVE", "NEGATIVE", "UNCERTAIN"}
VALID_TEMPORALITY  = {"CURRENT", "CLINICAL_HISTORY", "UPCOMING", "UNCERTAIN"}
VALID_SUBJECTS     = {"PATIENT", "FAMILY_MEMBER"}

# Document/UI noise patterns that are NOT medical entities
NOISE_PATTERNS = [
    r"page_no",
    r"\bip ccf\b",                     # document reference prefix
    r"discharge summary",
    r"discharge su\b",
    r"admit notice",
    r"info hub",
    r"quick search",
    r"fax inbox",
    r"inpatient records and er",
    r"\billumia\b",
    r"\bempatia\b",
    r"\bagmc\b",
    r"palliative care inpatient records",
    r"hp\b",                           # H&P document suffix
    r"micu hp",
    r"hem onc cons",
    r"cardio consu",
    r"critical car",
    r"infectious d",
    r"dc summary",
    r"\bask eva\b",
]

# Keywords strongly associated with each assertion type
ASSERTION_KEYWORDS = {
    "NEGATIVE": [
        "no ", "not ", "denies", "denied", "without", "negative",
        "absent", "never", "none", "ruled out", "free of",
    ],
    "UNCERTAIN": [
        "possible", "probable", "likely", "suspected", "rule out",
        "may", "might", "consider", "concern for", "cannot exclude",
        "uncertain", "unclear", "question of", "?",
    ],
}

# Keywords for temporality detection
TEMPORALITY_KEYWORDS = {
    "CLINICAL_HISTORY": [
        "history of", "h/o", "hx of", "past", "prior", "previous",
        "remote", "childhood", "years ago", "former",
    ],
    "UPCOMING": [
        "scheduled", "plan", "follow-up", "will", "upcoming",
        "appointment", "refer", "return", "next visit",
    ],
}


# ── API helper ────────────────────────────────────────────────────────────────

# OpenRouter model — swap to any model slug from https://openrouter.ai/models
# Good free/cheap options:
#   "google/gemini-2.0-flash-exp:free"   — Gemini 2.0 Flash (free tier)
#   "meta-llama/llama-3.3-70b-instruct"  — Llama 3.3 70B
#   "mistralai/mistral-small-3.1-24b-instruct:free" — Mistral Small (free tier)
#   "deepseek/deepseek-chat-v3-0324:free" — DeepSeek v3 (free tier)
OPENROUTER_MODEL   = "google/gemini-2.0-flash-exp:free"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "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"

def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
    """Call OpenRouter API; return text response or empty string on failure."""
    api_key = os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    if not api_key:
        print("[!] OPENROUTER_API_KEY not set — skipping LLM evaluation")
        return ""

    try:
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": OPENROUTER_MODEL,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            }),
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[!] OpenRouter API call failed: {e}")
        return ""


# ── Rule-based helpers ────────────────────────────────────────────────────────

def _is_noise_entity(entity: str, text: str, heading: str) -> bool:
    """Return True if the entity looks like OCR/UI noise rather than a clinical concept."""
    combined = (entity + " " + text + " " + heading).lower()
    return any(re.search(p, combined, re.IGNORECASE) for p in NOISE_PATTERNS)


def _detect_assertion_heuristic(entity: str, text: str) -> str | None:
    """Return NEGATIVE/UNCERTAIN/None based on keyword scan."""
    combined = (entity + " " + text).lower()
    for assertion, keywords in ASSERTION_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return assertion
    return None


def _detect_temporality_heuristic(text: str) -> str | None:
    combined = text.lower()
    for temporality, keywords in TEMPORALITY_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return temporality
    return None


def _check_family_subject(text: str) -> bool:
    """Return True if the text indicates a family member rather than the patient."""
    family_kws = [
        "father", "mother", "brother", "sister", "son", "daughter",
        "parent", "sibling", "uncle", "aunt", "grandfather", "grandmother",
        "family history", "fh:", "fh ",
    ]
    return any(kw in text.lower() for kw in family_kws)


def _metadata_completeness(metadata: dict) -> float:
    """Score metadata_from_qa completeness 0-1.
    An empty dict is 0.0. Non-empty is 1.0 (structure validated elsewhere)."""
    if not metadata:
        return 0.0
    # Penalise if expected keys are missing
    expected_keys = {"start_date", "end_date", "value", "unit", "route", "frequency"}
    present = len(set(metadata.keys()) & expected_keys)
    return min(1.0, present / max(1, len(expected_keys)) * 2)


def _has_event_date(metadata: dict, text: str) -> bool:
    """Check if a date is present in metadata or text."""
    if metadata.get("start_date") or metadata.get("event_date"):
        return True
    date_pattern = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
    return bool(re.search(date_pattern, text))


# ── Per-entity rule-based scoring ─────────────────────────────────────────────

def _score_entity_rules(entity: dict) -> dict:
    """
    Returns a dict with boolean error flags for each dimension.
    True = error detected.
    """
    ent_text    = entity.get("entity", "")
    ent_type    = entity.get("entity_type", "")
    assertion   = entity.get("assertion", "")
    temporality = entity.get("temporality", "")
    subject     = entity.get("subject", "")
    metadata    = entity.get("metadata_from_qa") or {}
    text        = entity.get("text", "")
    heading     = entity.get("heading", "")

    errors = {
        "entity_type_error":   False,
        "assertion_error":     False,
        "temporality_error":   False,
        "subject_error":       False,
        "event_date_missing":  False,
        "metadata_incomplete": False,
        "is_noise":            False,
    }

    # ── Noise detection ──────────────────────────────────────────────────────
    # NOTE: do NOT return early — noise entities still carry assertion/
    # temporality/subject labels that should be evaluated independently.
    if _is_noise_entity(ent_text, text, heading):
        errors["is_noise"] = True
        errors["entity_type_error"] = True
        # Noise entities should never be UNCERTAIN assertion — that's also wrong
        if assertion == "UNCERTAIN":
            errors["assertion_error"] = True

    # ── Entity type validation ───────────────────────────────────────────────
    elif ent_type not in VALID_ENTITY_TYPES:
        # Only flag type error here if not already flagged by noise check
        errors["entity_type_error"] = True

    # ── Assertion validation ─────────────────────────────────────────────────
    if assertion not in VALID_ASSERTIONS:
        errors["assertion_error"] = True
    else:
        heuristic = _detect_assertion_heuristic(ent_text, text)
        if heuristic and heuristic != assertion:
            errors["assertion_error"] = True

    # ── Temporality validation ───────────────────────────────────────────────
    if temporality not in VALID_TEMPORALITY:
        errors["temporality_error"] = True
    else:
        heuristic = _detect_temporality_heuristic(text)
        if heuristic and heuristic != temporality:
            errors["temporality_error"] = True

    # ── Subject validation ───────────────────────────────────────────────────
    if subject not in VALID_SUBJECTS:
        errors["subject_error"] = True
    elif subject == "PATIENT" and _check_family_subject(text):
        errors["subject_error"] = True

    # ── Event date / metadata ────────────────────────────────────────────────
    if not _has_event_date(metadata, text):
        errors["event_date_missing"] = True

    completeness = _metadata_completeness(metadata)
    if completeness < 0.5:
        errors["metadata_incomplete"] = True

    return errors


# ── LLM batch evaluation ──────────────────────────────────────────────────────

LLM_SYSTEM = """You are a clinical NLP evaluation expert auditing an AI extraction pipeline.

You are given:
  1. The ORIGINAL CLINICAL CHART (OCR markdown) — this is the GROUND TRUTH.
  2. A list of EXTRACTED ENTITIES the pipeline produced from that chart.

Your job: for each extracted entity, verify it against the ground truth chart and flag errors.

## Evaluation criteria

### entity_type_error = true if ANY of:
- The entity does not actually appear in the chart
- The entity is OCR noise, a UI element, document title, navigation text, or page metadata
- The entity is real but assigned the wrong category
  (e.g. a drug labelled PROBLEM, a lab result labelled PROCEDURE)

Valid types: MEDICINE, PROBLEM, PROCEDURE, TEST, VITAL_NAME, IMMUNIZATION,
             MEDICAL_DEVICE, MENTAL_STATUS, SDOH, SOCIAL_HISTORY

### assertion_error = true if ANY of:
- POSITIVE assigned but the chart says the condition is absent/denied/negated
- NEGATIVE assigned but the chart clearly confirms the condition is present
- UNCERTAIN assigned when the chart unambiguously confirms or denies the entity

### temporality_error = true if ANY of:
- CURRENT assigned but the chart places the entity in the patient's past history
- CLINICAL_HISTORY assigned but the chart describes it as an active/current issue
- UPCOMING assigned but there is no scheduled/planned event in the chart
- The entity falls under a section heading (e.g. "Past Medical History", "Plan")
  that clearly implies a different temporality than what was labelled

### subject_error = true if:
- PATIENT assigned but the entity comes from a Family History section or
  the text explicitly names a family member (father, mother, sibling, etc.)
- FAMILY_MEMBER assigned but the entity clearly refers to the patient

## Output format
Return ONLY valid JSON — no prose, no markdown fences:
{
  "evaluations": [
    {
      "idx": 0,
      "entity_type_error": false,
      "assertion_error": false,
      "temporality_error": false,
      "subject_error": false,
      "is_noise": false,
      "reasoning": "one-line reason citing the chart section or text span"
    }
  ]
}
"""

# Max characters of the chart MD to include per batch (fits in ~8k context comfortably)
MD_CONTEXT_LIMIT = 12_000

def _llm_evaluate_batch(entities: list[dict], chart_md: str = "") -> list[dict]:
    """
    Send a batch of entities to the LLM for evaluation against the chart ground truth.
    chart_md: full OCR markdown of the source clinical chart.
    """
    if not entities:
        return []

    # Trim entities to essential fields to save tokens
    trimmed = []
    for i, e in enumerate(entities):
        trimmed.append({
            "idx": i,
            "entity": e.get("entity", ""),
            "entity_type": e.get("entity_type", ""),
            "assertion": e.get("assertion", ""),
            "temporality": e.get("temporality", ""),
            "subject": e.get("subject", ""),
            # Include the pipeline's own text snippet as a hint, but the
            # LLM should verify against the full chart_md below
            "extracted_text_snippet": e.get("text", "")[:200],
            "heading": e.get("heading", "")[:150],
        })

    # Build user prompt: ground truth chart first, then entities to evaluate
    chart_section = ""
    if chart_md:
        truncated = chart_md[:MD_CONTEXT_LIMIT]
        if len(chart_md) > MD_CONTEXT_LIMIT:
            truncated += "\n\n[... chart truncated for context window ...]"
        chart_section = (
            "## GROUND TRUTH CLINICAL CHART (OCR markdown)\n"
            "```\n" + truncated + "\n```\n\n"
        )

    user_prompt = (
        chart_section
        + f"## EXTRACTED ENTITIES TO EVALUATE ({len(trimmed)} total)\n\n"
        + json.dumps(trimmed, indent=2)
    )

    response = _call_llm(LLM_SYSTEM, user_prompt, max_tokens=2000)

    if not response:
        return []

    # Parse JSON from response (strip markdown fences if model adds them)
    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        parsed = json.loads(clean)
        return parsed.get("evaluations", [])
    except Exception as e:
        print(f"[!] Failed to parse LLM response: {e}")
        return []


# ── Main evaluator class ──────────────────────────────────────────────────────

class ClinicalEntityEvaluator:
    """
    Evaluates a list of clinical entities and returns the standard output schema.
    Uses rule-based checks + optional LLM refinement.
    """

    def __init__(self, use_llm: bool = True, llm_batch_size: int = 20):
        self.use_llm = use_llm  # key falls back to hardcoded default; always available
        self.llm_batch_size = llm_batch_size

    def evaluate(self, file_name: str, entities: list[dict], chart_md: str = "") -> dict:
        if not isinstance(entities, list):
            # Some files wrap entities under a key
            if isinstance(entities, dict):
                entities = entities.get("entities", list(entities.values())[0] if entities else [])

        # ── Step 1: Rule-based pass ──────────────────────────────────────────
        rule_scores = [_score_entity_rules(e) for e in entities]

        # ── Step 2: LLM refinement ───────────────────────────────────────────
        llm_scores_map: dict[int, dict] = {}
        if self.use_llm:
            # Only send non-trivially-noise entities to LLM (save tokens)
            candidate_indices = [
                i for i, s in enumerate(rule_scores) if not s.get("is_noise")
            ]
            for batch_start in range(0, len(candidate_indices), self.llm_batch_size):
                batch_idx = candidate_indices[batch_start: batch_start + self.llm_batch_size]
                batch_entities = [entities[i] for i in batch_idx]
                llm_results = _llm_evaluate_batch(batch_entities, chart_md=chart_md)
                for llm_r in llm_results:
                    local_idx = llm_r.get("idx", -1)
                    if 0 <= local_idx < len(batch_idx):
                        global_idx = batch_idx[local_idx]
                        llm_scores_map[global_idx] = llm_r
                time.sleep(0.3)  # gentle rate-limiting

        # ── Step 3: Merge scores ─────────────────────────────────────────────
        merged = []
        for i, (entity, rule) in enumerate(zip(entities, rule_scores)):
            llm = llm_scores_map.get(i, {})
            merged.append({
                "entity_type":       entity.get("entity_type", ""),
                "assertion":         entity.get("assertion", ""),
                "temporality":       entity.get("temporality", ""),
                "subject":           entity.get("subject", ""),
                # OR: flag if either rule OR LLM found an error
                "entity_type_error": rule["entity_type_error"] or llm.get("entity_type_error", False),
                "assertion_error":   rule["assertion_error"]   or llm.get("assertion_error", False),
                "temporality_error": rule["temporality_error"] or llm.get("temporality_error", False),
                "subject_error":     rule["subject_error"]     or llm.get("subject_error", False),
                "event_date_missing":rule["event_date_missing"],
                "metadata_incomplete": rule["metadata_incomplete"],
            })

        # ── Step 4: Aggregate rates ──────────────────────────────────────────
        return self._aggregate(file_name, merged, entities)

    def _aggregate(self, file_name: str, merged: list[dict], entities: list[dict]) -> dict:
        n = len(merged)
        if n == 0:
            return self._empty_output(file_name)

        # Entity type error rates per category.
        # Bucket by claimed type. Categories absent from this file → None (not 0.0).
        # Entities with invalid types go into WRONG_TYPE bucket.
        entity_type_errors = defaultdict(lambda: {"total": 0, "errors": 0})
        for m in merged:
            et = m["entity_type"] if m["entity_type"] in VALID_ENTITY_TYPES else "WRONG_TYPE"
            entity_type_errors[et]["total"] += 1
            if m["entity_type_error"]:
                entity_type_errors[et]["errors"] += 1

        entity_type_error_rate = {}
        for et in [
            "MEDICINE", "PROBLEM", "PROCEDURE", "TEST", "VITAL_NAME",
            "IMMUNIZATION", "MEDICAL_DEVICE", "MENTAL_STATUS", "SDOH", "SOCIAL_HISTORY",
        ]:
            stats = entity_type_errors[et]
            total = stats["total"]
            # None = category not present in this file; avoids false "0.0 = no errors"
            entity_type_error_rate[et] = round(stats["errors"] / total, 4) if total > 0 else None

        # Overall entity error rate across ALL types (always computable)
        total_entity_errors = sum(1 for m in merged if m["entity_type_error"])
        entity_type_error_rate["_overall"] = round(total_entity_errors / n, 4)

        # Assertion error rates — None if that assertion value never appears in file
        assertion_errors = defaultdict(lambda: {"total": 0, "errors": 0})
        for m in merged:
            av = m["assertion"] if m["assertion"] in VALID_ASSERTIONS else "WRONG_ASSERTION"
            assertion_errors[av]["total"] += 1
            if m["assertion_error"]:
                assertion_errors[av]["errors"] += 1

        assertion_error_rate = {}
        for av in ["POSITIVE", "NEGATIVE", "UNCERTAIN"]:
            stats = assertion_errors[av]
            total = stats["total"]
            assertion_error_rate[av] = round(stats["errors"] / total, 4) if total > 0 else None
        total_assertion_errors = sum(1 for m in merged if m["assertion_error"])
        assertion_error_rate["_overall"] = round(total_assertion_errors / n, 4)

        # Temporality error rates — None if not present in file
        temp_errors = defaultdict(lambda: {"total": 0, "errors": 0})
        for m in merged:
            tv = m["temporality"] if m["temporality"] in VALID_TEMPORALITY else "WRONG_TEMPORALITY"
            temp_errors[tv]["total"] += 1
            if m["temporality_error"]:
                temp_errors[tv]["errors"] += 1

        temporality_error_rate = {}
        for tv in ["CURRENT", "CLINICAL_HISTORY", "UPCOMING", "UNCERTAIN"]:
            stats = temp_errors[tv]
            total = stats["total"]
            temporality_error_rate[tv] = round(stats["errors"] / total, 4) if total > 0 else None
        total_temp_errors = sum(1 for m in merged if m["temporality_error"])
        temporality_error_rate["_overall"] = round(total_temp_errors / n, 4)

        # Subject error rates — None if not present in file
        subj_errors = defaultdict(lambda: {"total": 0, "errors": 0})
        for m in merged:
            sv = m["subject"] if m["subject"] in VALID_SUBJECTS else "WRONG_SUBJECT"
            subj_errors[sv]["total"] += 1
            if m["subject_error"]:
                subj_errors[sv]["errors"] += 1

        subject_error_rate = {}
        for sv in ["PATIENT", "FAMILY_MEMBER"]:
            stats = subj_errors[sv]
            total = stats["total"]
            subject_error_rate[sv] = round(stats["errors"] / total, 4) if total > 0 else None
        total_subj_errors = sum(1 for m in merged if m["subject_error"])
        subject_error_rate["_overall"] = round(total_subj_errors / n, 4)

        # Event date accuracy (fraction with a date present)
        date_present = sum(1 for m in merged if not m["event_date_missing"])
        event_date_accuracy = round(date_present / n, 4)

        # Attribute completeness
        completeness_scores = []
        for e in entities:
            md = e.get("metadata_from_qa") or {}
            completeness_scores.append(_metadata_completeness(md))
        attribute_completeness = round(sum(completeness_scores) / n, 4)

        return {
            "file_name": file_name,
            "entity_type_error_rate": entity_type_error_rate,
            "assertion_error_rate": assertion_error_rate,
            "temporality_error_rate": temporality_error_rate,
            "subject_error_rate": subject_error_rate,
            "event_date_accuracy": event_date_accuracy,
            "attribute_completeness": attribute_completeness,
        }

    def _empty_output(self, file_name: str) -> dict:
        return {
            "file_name": file_name,
            "entity_type_error_rate": {k: 0.0 for k in VALID_ENTITY_TYPES},
            "assertion_error_rate": {k: 0.0 for k in VALID_ASSERTIONS},
            "temporality_error_rate": {k: 0.0 for k in VALID_TEMPORALITY},
            "subject_error_rate": {k: 0.0 for k in VALID_SUBJECTS},
            "event_date_accuracy": 0.0,
            "attribute_completeness": 0.0,
        }