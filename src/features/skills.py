"""Dictionary-based skill extractor with longest-alias-first matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.paths import ALIAS_POLICY_PATH, SKILLS_PATH
from src.preprocess.text import normalize_text

# Family hiển thị trên UI / form. Domain & industrial không gắn vào CV.
COMMON_FAMILIES = {
    "language",
    "frontend",
    "backend",
    "mobile",
    "data",
    "ai_ml",
    "devops",
    "cloud",
    "qa",
    "security",
}
SKIP_FAMILIES = {"soft", "domain", "other", "industrial"}

DOMAIN_SKILLS = {
    "marketplace",
    "logistics",
    "ecommerce",
    "fintech",
    "healthtech",
    "edtech",
    "hrtech",
    "adtech",
    "gaming",
    "growth",
}
INDUSTRIAL_SKILLS = {
    "cad",
    "cam",
    "plc",
    "modbus",
    "canbus",
    "autosar",
    "altium",
    "ansys",
    "abaqus",
    "comsol",
    "labview",
    "fpga",
    "asic",
    "vhdl",
    "verilog_extra",
    "uvm",
}

# Công cụ design gắn nhầm family frontend — không tính là skill IT khi lọc corpus.
DESIGN_SKILLS = {
    "photoshop",
    "figma",
    "figma_devmode",
    "uiux",
    "premiere",
    "canva",
    "after_effects",
    "blender_3d",
    "cinema4d",
    "maya",
    "principle",
    "zeplin",
    "invision",
    "zbrush",
    "substance",
}

# Alias quá rộng / tiếng Việt trùng / 2-3 ký tự mơ hồ.
ALIAS_BLOCKLIST = {
    "ui",
    "ux",
    "cv",
    "rest",
    "json",
    "xml",
    "ba",
    "po",
    "pm",
    "ml",
    "dl",
    "tf",
    "ar",
    "fc",
    "cli",
    "cdn",
    "csv",
    "san",
    "nas",
    "raid",
    "node",
    "game",
    "scripting",
    "embedded",
    "cnn",
    "retry",
    "proxy",
    "quay",  # trùng tiếng Việt "quay"
    "cam",
    "cad",
    "a b",
    "a/b",
    "js",  # dễ dính trong từ; giữ javascript/typescript đầy đủ
}

# Token ngắn nhưng đủ đặc trưng.
ALIAS_ALLOW_SHORT = {
    "c#",
    "c++",
    "sql",
    "css",
    "aws",
    "gcp",
    "k8s",
    "s3",
    "php",
    "cpp",
    "qt",
    "k6",
    "d3",
    "git",
    "vue",
    "ios",
    "go",
    "r",
    "sso",
    "nlp",
    "etl",
    "bia",
    "iam",
    "vpc",
    "ecs",
    "eks",
    "ec2",
    "emr",
    "dbt",
    "ci",
    "qa",
}

MIN_ALIAS_LEN = 3


@dataclass
class SkillTaxonomy:
    canonical_to_family: dict[str, str]
    alias_to_canonical: dict[str, str]
    aliases_by_length: list[str]
    patterns: list[tuple[re.Pattern, str]] = field(default_factory=list)

    def all_canonical(self) -> list[str]:
        return sorted(self.canonical_to_family.keys())

    def families(self) -> list[str]:
        return sorted(set(self.canonical_to_family.values()) - SKIP_FAMILIES)


def _family_for(canon: str, raw_family: str) -> str:
    if canon in DOMAIN_SKILLS:
        return "domain"
    if canon in INDUSTRIAL_SKILLS:
        return "industrial"
    fam = (raw_family or "backend").strip().lower()
    if fam in SKIP_FAMILIES:
        return fam
    if fam in COMMON_FAMILIES:
        return fam
    if fam in {"infra", "sre", "sysadmin", "network"}:
        return "devops"
    if fam in {"ml", "ai", "genai"}:
        return "ai_ml"
    if fam in {"web"}:
        return "frontend"
    return "backend"


def _load_alias_policy(policy_path: Path | None) -> dict:
    path = policy_path if policy_path is not None else ALIAS_POLICY_PATH
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_alias_policy(
    canonical_to_family: dict[str, str],
    alias_to_canonical: dict[str, str],
    policy: dict,
) -> tuple[dict[str, str], dict[str, str]]:
    extra_block = {_norm_alias(str(a)) for a in (policy.get("block_aliases") or []) if a}
    for alias in list(alias_to_canonical):
        if alias in extra_block:
            alias_to_canonical.pop(alias, None)
    drop = policy.get("drop_aliases") or {}
    for canon, aliases in drop.items():
        for raw in aliases or []:
            an = _norm_alias(str(raw))
            if alias_to_canonical.get(an) == canon:
                alias_to_canonical.pop(an, None)
    merge = policy.get("merge_into") or {}
    for src, dst in merge.items():
        if not src or not dst or src == dst:
            continue
        if dst not in canonical_to_family and src in canonical_to_family:
            canonical_to_family[dst] = canonical_to_family[src]
        for alias, canon in list(alias_to_canonical.items()):
            if canon == src:
                alias_to_canonical[alias] = dst
        canonical_to_family.pop(src, None)
    return canonical_to_family, alias_to_canonical


def load_taxonomy(path: Path | None = None, apply_policy: bool | None = None) -> SkillTaxonomy:
    path = path or SKILLS_PATH
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    skills = data.get("skills") or {}
    canonical_to_family: dict[str, str] = {}
    alias_to_canonical: dict[str, str] = {}
    for canon, meta in skills.items():
        family = _family_for(canon, (meta or {}).get("family") or "other")
        canonical_to_family[canon] = family
        raw_aliases = (meta or {}).get("aliases") or []
        if not isinstance(raw_aliases, list):
            raw_aliases = [raw_aliases]
        auto = canon.replace("_", " ")
        names = []
        if _alias_ok(auto):
            names.append(auto)
        names.extend(str(a) for a in raw_aliases if a)
        for alias in names:
            alias_n = _norm_alias(alias)
            if not _alias_ok(alias_n):
                continue
            alias_to_canonical.setdefault(alias_n, canon)
    if apply_policy is None:
        apply_policy = ALIAS_POLICY_PATH.exists()
    if apply_policy:
        canonical_to_family, alias_to_canonical = _apply_alias_policy(
            canonical_to_family, alias_to_canonical, _load_alias_policy(None)
        )
    aliases_by_length = sorted(alias_to_canonical.keys(), key=len, reverse=True)
    patterns: list[tuple[re.Pattern, str]] = []
    for alias in aliases_by_length:
        canon = alias_to_canonical[alias]
        if canon not in canonical_to_family:
            continue
        patterns.append((_alias_pattern(alias), canon))
    return SkillTaxonomy(
        canonical_to_family=canonical_to_family,
        alias_to_canonical=alias_to_canonical,
        aliases_by_length=aliases_by_length,
        patterns=patterns,
    )


def _norm_alias(alias: str) -> str:
    return re.sub(r"\s+", " ", (alias or "").strip().lower())


def _alias_ok(alias: str) -> bool:
    if not alias:
        return False
    if alias in ALIAS_BLOCKLIST:
        return False
    compact = re.sub(r"[^a-z0-9+#]", "", alias)
    if alias in ALIAS_ALLOW_SHORT or compact in ALIAS_ALLOW_SHORT:
        return True
    if len(compact) < MIN_ALIAS_LEN:
        return False
    return True


def _alias_pattern(alias: str) -> re.Pattern:
    """Biên giới từ Unicode: không khớp 'ui' trong 'tuổi', 'quay' trong câu tiếng Việt.

    Khoảng trắng trong alias chỉ được thay bằng 1 separator (c/c++, ci-cd), không
    được `*` xuyên cả document.
    """
    escaped = re.escape(alias)
    escaped = escaped.replace(r"\ ", r"[\s\-_/\.]+")
    if alias == "java":
        return re.compile(r"(?<![\w])java(?!script)(?![\w])", re.I)
    if alias in {"c#", "c++"}:
        return re.compile(rf"(?<![\w]){escaped}(?![\w])", re.I)
    return re.compile(rf"(?<![\w]){escaped}(?![\w])", re.I)


class SkillExtractor:
    def __init__(self, taxonomy: SkillTaxonomy | None = None) -> None:
        self.taxonomy = taxonomy or load_taxonomy()

    def extract(
        self,
        text: str,
        extra_tags: list[str] | None = None,
        include_soft: bool = False,
        order_by_position: bool = False,
    ) -> list[str]:
        blob = normalize_text(text).lower()
        if extra_tags:
            blob = blob + " " + " ".join(str(t) for t in extra_tags).lower()
        earliest: dict[str, int] = {}
        found: list[str] = []
        for pattern, canon in self.taxonomy.patterns:
            family = self.taxonomy.canonical_to_family.get(canon, "")
            if family in SKIP_FAMILIES and not include_soft:
                continue
            match = pattern.search(blob)
            if not match:
                continue
            pos = int(match.start())
            if order_by_position:
                if canon not in earliest or pos < earliest[canon]:
                    earliest[canon] = pos
                continue
            if canon in earliest:
                continue
            earliest[canon] = pos
            found.append(canon)
        if order_by_position:
            return [canon for canon, _ in sorted(earliest.items(), key=lambda kv: kv[1])]
        return found

    def family_ratio(self, skills: list[str]) -> dict[str, float]:
        families = self.taxonomy.families()
        counts = {f: 0.0 for f in families}
        for sk in skills:
            fam = self.taxonomy.canonical_to_family.get(sk)
            if fam and fam in counts:
                counts[fam] += 1
        total = sum(counts.values()) or 1.0
        return {f: counts[f] / total for f in families}
