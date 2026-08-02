"""cyber_sources.py — public cybersecurity sources for the AION-Cyber corpus.

Every source here is public and redistributable, and every document carries the
licence it came under.  That is not paperwork: a corpus whose provenance cannot
be stated cannot be published, audited, or defended, and AION's first operating
invariant is that we measure and record rather than claim.

Sources
-------
=========================  ==========================  ========================
Source                     What it teaches             Licence
=========================  ==========================  ========================
NVD CVE (NIST)             How a vulnerability is      US Government work,
                           described and scored        public domain
CISA KEV catalogue         Which flaws are actually    US Government work,
                           exploited, and the required public domain
                           response
MITRE ATT&CK               Adversary tactics and       ATT&CK Terms of Use,
                           techniques, and how they    free with attribution
                           are detected and mitigated
OWASP Top 10               How application risk is     CC BY-SA 4.0
                           reasoned about
OWASP Cheat Sheet Series   Concrete defensive          CC BY-SA 4.0
                           guidance
=========================  ==========================  ========================

Rendering, not dumping
----------------------
None of these arrive as prose.  They arrive as JSON records and STIX objects,
and a language model trained on raw JSON learns JSON.  Each fetcher therefore
*renders* its records into a consistent, labelled prose form — the same shape a
security analyst would write.  The structure is the lesson: the model should
learn that a vulnerability has a summary, a severity, an attack vector, and a
weakness class, and that those parts relate to each other.

Politeness
----------
NVD asks for roughly six seconds between anonymous requests, and this module
obeys that.  It is not optional courtesy; NVD throttles hard, and a corpus
build that gets a source's IP blocked is a corpus build that cannot be
repeated.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator

USER_AGENT = "AION-corpus-builder/0.1 (research corpus; contact via repository)"

# NVD's documented anonymous rate limit is 5 requests per rolling 30 seconds.
NVD_DELAY_SECONDS = 6.5
NVD_PAGE_SIZE = 2000          # NVD's maximum
GITHUB_DELAY_SECONDS = 0.4


# ── document envelope ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CyberDocument:
    """One corpus document with the provenance needed to defend its presence.

    ``provenance`` is either ``"genuine"`` (fetched from a real source) or
    ``"synthetic"`` (generated).  This module only ever produces ``genuine``;
    :mod:`cyber_reasoning` only ever produces ``synthetic``.  Nothing merges
    the two labels, so a corpus report can always say how much of it is real.
    """
    text: str
    source: str
    licence: str
    url: str
    category: str
    provenance: str = "genuine"


@dataclass
class SourceResult:
    """What one source contributed, and what went wrong if anything did."""
    name: str
    documents: list[CyberDocument] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.documents)


# ── HTTP ──────────────────────────────────────────────────────────────────────

def fetch_bytes(url: str, timeout: int = 60, retries: int = 3) -> bytes | None:
    """GET ``url`` with linear backoff.  Returns ``None`` when it never lands.

    Returning ``None`` rather than raising is deliberate: one unreachable
    source must degrade the corpus, not abort a build that has already spent
    twenty minutes on the others.  The caller records the failure and the
    build report shows it.
    """
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            if attempt == retries:
                print(f"    [warn] {url} failed after {retries} attempts: {exc}")
                return None
            time.sleep(2.0 * attempt)
    return None


def fetch_json(url: str, timeout: int = 60, retries: int = 3):
    raw = fetch_bytes(url, timeout=timeout, retries=retries)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        print(f"    [warn] {url} returned invalid JSON: {exc}")
        return None


def github_markdown_paths(repo: str, predicate: Callable[[str], bool]) -> list[str]:
    """List blob paths in ``repo`` matching ``predicate``, trying master then main.

    Listing the tree instead of hardcoding filenames means the fetcher keeps
    working when upstream adds a document — which for the Cheat Sheet Series is
    a monthly event.
    """
    for branch in ("master", "main"):
        data = fetch_json(
            f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1")
        if data and "tree" in data:
            return sorted(
                entry["path"] for entry in data["tree"]
                if entry.get("type") == "blob" and predicate(entry["path"])
            )
    return []


# ── NVD: CVE descriptions ─────────────────────────────────────────────────────

_CVSS_KEYS = ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2")

_VECTOR_WORDS = {
    "NETWORK": "remotely over a network",
    "ADJACENT_NETWORK": "from an adjacent network",
    "LOCAL": "with local access",
    "PHYSICAL": "with physical access",
}


def _cvss_summary(metrics: dict) -> tuple[str, str]:
    """Return (severity line, exploitability sentence) from an NVD metrics block."""
    for key in _CVSS_KEYS:
        entries = metrics.get(key) or []
        if not entries:
            continue
        data = entries[0].get("cvssData", {})
        score = data.get("baseScore")
        severity = data.get("baseSeverity") or entries[0].get("baseSeverity") or "UNKNOWN"
        severity_line = f"Severity: {str(severity).title()} (CVSS {score})" if score is not None else ""

        vector = _VECTOR_WORDS.get(str(data.get("accessVector") or data.get("attackVector", "")).upper())
        privileges = str(data.get("privilegesRequired", "")).upper()
        interaction = str(data.get("userInteraction", "")).upper()
        parts = []
        if vector:
            parts.append(f"exploitable {vector}")
        if privileges == "NONE":
            parts.append("without any privileges")
        elif privileges in ("LOW", "HIGH"):
            parts.append(f"with {privileges.lower()} privileges")
        if interaction == "NONE":
            parts.append("without user interaction")
        elif interaction == "REQUIRED":
            parts.append("only if a user is tricked into acting")
        exploit_line = ("Exploitability: " + ", ".join(parts) + ".") if parts else ""
        return severity_line, exploit_line
    return "", ""


def _render_cve(record: dict) -> str | None:
    cve = record.get("cve", {})
    cve_id = cve.get("id")
    if not cve_id:
        return None

    description = ""
    for entry in cve.get("descriptions", []):
        if entry.get("lang") == "en":
            description = (entry.get("value") or "").strip()
            break
    # Rejected or disputed entries carry no analytical content.
    if not description or description.startswith("** REJECT"):
        return None

    lines = [f"Vulnerability report: {cve_id}"]
    published = (cve.get("published") or "")[:10]
    if published:
        lines.append(f"Published: {published}")
    lines.append(f"Summary: {description}")

    severity_line, exploit_line = _cvss_summary(cve.get("metrics", {}))
    if severity_line:
        lines.append(severity_line)
    if exploit_line:
        lines.append(exploit_line)

    weaknesses = []
    for weakness in cve.get("weaknesses", []):
        for entry in weakness.get("description", []):
            value = entry.get("value", "")
            if value.startswith("CWE-") and value not in weaknesses:
                weaknesses.append(value)
    if weaknesses:
        lines.append("Weakness class: " + ", ".join(weaknesses[:4]))

    vendors = []
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                parts = (match.get("criteria") or "").split(":")
                if len(parts) > 5:
                    label = f"{parts[3]} {parts[4]}".replace("_", " ").strip()
                    if label and label not in vendors:
                        vendors.append(label)
    if vendors:
        lines.append("Affected products: " + ", ".join(vendors[:6]))

    return "\n".join(lines)


def fetch_nvd_cves(max_records: int) -> SourceResult:
    """Page through the NVD CVE API and render each entry as an advisory."""
    result = SourceResult(name="nvd-cve")
    if max_records <= 0:
        return result

    start_index = 0
    total = None
    while len(result.documents) < max_records:
        page_size = min(NVD_PAGE_SIZE, max_records - len(result.documents))
        url = ("https://services.nvd.nist.gov/rest/json/cves/2.0"
               f"?resultsPerPage={page_size}&startIndex={start_index}")
        payload = fetch_json(url, timeout=120)
        if payload is None:
            result.errors.append(f"NVD page at startIndex={start_index} failed")
            break

        if total is None:
            total = payload.get("totalResults", 0)
            print(f"    NVD reports {total:,} CVEs; requesting {max_records:,}")

        vulnerabilities = payload.get("vulnerabilities", [])
        if not vulnerabilities:
            break

        for record in vulnerabilities:
            rendered = _render_cve(record)
            if rendered:
                cve_id = record["cve"]["id"]
                result.documents.append(CyberDocument(
                    text=rendered,
                    source="NVD (NIST National Vulnerability Database)",
                    licence="US Government work — public domain",
                    url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                    category="cve",
                ))

        start_index += len(vulnerabilities)
        print(f"    fetched {len(result.documents):,} CVE descriptions", end="\r", flush=True)
        if total is not None and start_index >= total:
            break
        time.sleep(NVD_DELAY_SECONDS)

    print(f"    fetched {len(result.documents):,} CVE descriptions          ")
    return result


# ── CISA: Known Exploited Vulnerabilities ─────────────────────────────────────

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def fetch_cisa_kev() -> SourceResult:
    """CISA's catalogue of vulnerabilities under active exploitation.

    The highest-signal source in the set: every entry is a flaw attackers are
    provably using, paired with the action a defender must take.  That
    problem-to-required-action pairing is exactly the reasoning shape AION
    should learn.
    """
    result = SourceResult(name="cisa-kev")
    payload = fetch_json(KEV_URL, timeout=120)
    if payload is None:
        result.errors.append("CISA KEV catalogue unreachable")
        return result

    catalogue_version = payload.get("catalogVersion", "unknown")
    for entry in payload.get("vulnerabilities", []):
        cve_id = entry.get("cveID")
        if not cve_id:
            continue
        lines = [
            f"Known exploited vulnerability: {cve_id}",
            f"Name: {entry.get('vulnerabilityName', '').strip()}",
            f"Vendor and product: {entry.get('vendorProject', '')} {entry.get('product', '')}".strip(),
            f"Added to the CISA catalogue: {entry.get('dateAdded', '')}",
            f"Description: {(entry.get('shortDescription') or '').strip()}",
            f"Required action: {(entry.get('requiredAction') or '').strip()}",
            f"Action due by: {entry.get('dueDate', '')}",
        ]
        ransomware = (entry.get("knownRansomwareCampaignUse") or "Unknown").strip()
        lines.append(f"Known use in ransomware campaigns: {ransomware}")
        if entry.get("notes"):
            lines.append(f"Notes: {entry['notes'].strip()}")
        lines.append(
            "Why this matters: a vulnerability in this catalogue is not "
            "theoretical. It is being exploited against real organisations, so "
            "remediation is time-bound rather than discretionary."
        )
        result.documents.append(CyberDocument(
            text="\n".join(line for line in lines if line.strip().rstrip(":")),
            source=f"CISA Known Exploited Vulnerabilities catalogue {catalogue_version}",
            licence="US Government work — public domain",
            url="https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
            category="advisory",
        ))
    return result


# ── MITRE ATT&CK ──────────────────────────────────────────────────────────────

ATTACK_BUNDLES = {
    "enterprise": "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json",
    "mobile": "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/mobile-attack/mobile-attack.json",
}


def _attack_id(obj: dict) -> str:
    for ref in obj.get("external_references", []):
        if ref.get("source_name", "").endswith("attack"):
            return ref.get("external_id", "")
    return ""


def _render_attack_technique(obj: dict, mitigations: dict[str, list[str]]) -> str | None:
    technique_id = _attack_id(obj)
    name = obj.get("name")
    description = (obj.get("description") or "").strip()
    if not technique_id or not name or not description:
        return None

    tactics = [phase.get("phase_name", "").replace("-", " ")
               for phase in obj.get("kill_chain_phases", [])]
    lines = [f"Adversary technique: {technique_id} — {name}"]
    if tactics:
        lines.append("Tactic: " + ", ".join(sorted(set(tactics))))
    platforms = obj.get("x_mitre_platforms") or []
    if platforms:
        lines.append("Platforms: " + ", ".join(platforms))
    lines.append(f"Description: {description}")

    detection = (obj.get("x_mitre_detection") or "").strip()
    if detection:
        lines.append(f"Detection: {detection}")
    applied = mitigations.get(obj.get("id", ""))
    if applied:
        lines.append("Mitigations: " + "; ".join(applied[:6]))
    return "\n".join(lines)


def fetch_mitre_attack(domains: tuple[str, ...] = ("enterprise", "mobile")) -> SourceResult:
    """Techniques, tactics, mitigations, and threat-group profiles from ATT&CK.

    ATT&CK is the closest thing the field has to a shared vocabulary for
    adversary behaviour.  Training on it is what lets a model name what it is
    looking at instead of only saying "this looks bad".
    """
    result = SourceResult(name="mitre-attack")
    for domain in domains:
        url = ATTACK_BUNDLES.get(domain)
        if not url:
            continue
        print(f"    downloading ATT&CK {domain} bundle (large, please wait)...")
        bundle = fetch_json(url, timeout=300)
        if bundle is None:
            result.errors.append(f"ATT&CK {domain} bundle unreachable")
            continue

        objects = bundle.get("objects", [])
        # Mitigation objects are joined to techniques by `mitigates` relationships.
        mitigation_names = {
            obj["id"]: obj.get("name", "")
            for obj in objects
            if obj.get("type") == "course-of-action"
        }
        applied: dict[str, list[str]] = {}
        for obj in objects:
            if obj.get("type") == "relationship" and obj.get("relationship_type") == "mitigates":
                name = mitigation_names.get(obj.get("source_ref", ""))
                if name:
                    applied.setdefault(obj.get("target_ref", ""), []).append(name)

        for obj in objects:
            if obj.get("x_mitre_deprecated") or obj.get("revoked"):
                continue
            kind = obj.get("type")
            rendered = None
            category = "attack-technique"

            if kind == "attack-pattern":
                rendered = _render_attack_technique(obj, applied)
            elif kind == "intrusion-set" and (obj.get("description") or "").strip():
                group_id = _attack_id(obj)
                aliases = [a for a in (obj.get("aliases") or []) if a != obj.get("name")]
                rendered = "\n".join(filter(None, [
                    f"Threat group: {group_id} — {obj.get('name')}",
                    ("Also known as: " + ", ".join(aliases)) if aliases else "",
                    f"Description: {obj['description'].strip()}",
                ]))
                category = "threat-actor"
            elif kind == "course-of-action" and (obj.get("description") or "").strip():
                rendered = "\n".join([
                    f"Mitigation: {_attack_id(obj)} — {obj.get('name')}",
                    f"Description: {obj['description'].strip()}",
                ])
                category = "mitigation"

            if rendered:
                result.documents.append(CyberDocument(
                    text=rendered,
                    source=f"MITRE ATT&CK ({domain})",
                    licence="MITRE ATT&CK Terms of Use — free use with attribution",
                    url="https://attack.mitre.org/",
                    category=category,
                ))
    return result


# ── OWASP ─────────────────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Reduce Markdown to the prose a reader would actually take in.

    Fenced code blocks, tables, and badge images are removed: they are a large
    fraction of these files by bytes and near-zero by teaching value for a
    model this size, and code fences in particular would teach it to emit
    fences in the middle of an explanation.
    """
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("|") or set(stripped) <= {"-", "|", ":", " "} and stripped:
            continue
        if stripped.startswith(("![", "[![")):
            continue
        if stripped.startswith("#"):
            out.append(stripped.lstrip("#").strip())
            continue
        out.append(line)
    return "\n".join(out)


def fetch_owasp(include_cheatsheets: bool = True) -> SourceResult:
    """OWASP Top 10 (English) and, optionally, the Cheat Sheet Series."""
    result = SourceResult(name="owasp")

    top10 = github_markdown_paths(
        "OWASP/Top10",
        lambda p: p.startswith("2021/docs/en/") and p.endswith(".md"),
    )
    if not top10:
        result.errors.append("OWASP Top 10 file listing unavailable")
    for path in top10:
        raw = fetch_bytes(f"https://raw.githubusercontent.com/OWASP/Top10/master/{path}")
        if raw is None:
            continue
        result.documents.append(CyberDocument(
            text=_strip_markdown(raw.decode("utf-8", errors="replace")),
            source="OWASP Top 10 (2021)",
            licence="CC BY-SA 4.0",
            url=f"https://github.com/OWASP/Top10/blob/master/{path}",
            category="web-security",
        ))
        time.sleep(GITHUB_DELAY_SECONDS)

    if include_cheatsheets:
        sheets = github_markdown_paths(
            "OWASP/CheatSheetSeries",
            lambda p: p.startswith("cheatsheets/") and p.endswith(".md"),
        )
        if not sheets:
            result.errors.append("OWASP Cheat Sheet listing unavailable")
        for index, path in enumerate(sheets, 1):
            raw = fetch_bytes(
                f"https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/{path}")
            if raw is None:
                continue
            result.documents.append(CyberDocument(
                text=_strip_markdown(raw.decode("utf-8", errors="replace")),
                source="OWASP Cheat Sheet Series",
                licence="CC BY-SA 4.0",
                url=f"https://github.com/OWASP/CheatSheetSeries/blob/master/{path}",
                category="defensive-guidance",
            ))
            print(f"    OWASP cheat sheets {index}/{len(sheets)}", end="\r", flush=True)
            time.sleep(GITHUB_DELAY_SECONDS)
        print(" " * 48, end="\r")

    return result


# ── registry ──────────────────────────────────────────────────────────────────

def all_sources(max_cves: int, include_cheatsheets: bool = True
                ) -> Iterator[Callable[[], SourceResult]]:
    """The genuine-source fetchers, cheapest first.

    Ordered so a build that is interrupted or rate-limited has already banked
    the small high-value sources before spending twenty minutes inside NVD.
    """
    yield lambda: fetch_cisa_kev()
    yield lambda: fetch_owasp(include_cheatsheets=include_cheatsheets)
    yield lambda: fetch_mitre_attack()
    yield lambda: fetch_nvd_cves(max_cves)
