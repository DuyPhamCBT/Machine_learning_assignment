"""Role family from job title (rule-based, used as weak label — not a train target)."""

from __future__ import annotations

import re

RULES: list[tuple[str, list[str]]] = [
    ("AI/ML", [r"\bml\b", r"machine learning", r"deep learning", r"\bai\b", r"llm", r"nlp", r"computer vision", r"data scientist", r"genai", r"generative"]),
    ("Data", [r"data engineer", r"data analyst", r"business intelligence", r"\bbi\b", r"etl", r"warehouse", r"analytics engineer"]),
    ("DevOps", [r"devops", r"\bsre\b", r"platform engineer", r"cloud engineer", r"site reliability"]),
    ("Security", [r"security", r"cyber", r"soc analyst", r"pentester", r"appsec"]),
    ("QA", [r"\bqa\b", r"\bqc\b", r"tester", r"sdet", r"automation test", r"quality"]),
    ("Mobile", [r"mobile", r"android", r"ios", r"flutter", r"react native"]),
    ("Fullstack", [r"fullstack", r"full-stack", r"full stack"]),
    ("Frontend", [r"frontend", r"front-end", r"front end", r"react", r"vue", r"angular", r"ui engineer"]),
    ("Backend", [r"backend", r"back-end", r"back end", r"java", r"spring", r"django", r"fastapi", r"nestjs", r"golang", r"\.net"]),
    ("PM/BA", [r"business analyst", r"\bba\b", r"product owner", r"product manager", r"\bpm\b", r"scrum master"]),
]


def infer_role_family(title: str | None, skills: list[str] | None = None) -> str:
    blob = (title or "").lower()
    skill_blob = " ".join(skills or []).lower()
    text = f"{blob} {skill_blob}"
    for family, patterns in RULES:
        for pat in patterns:
            if re.search(pat, text, re.I):
                return family
    if re.search(r"developer|engineer|programmer|lập trình", blob):
        return "Backend"
    return "Other"
