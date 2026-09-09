"""CourtListener bulk-data ingestion — the primary ingestion path.

Why bulk data instead of the live REST API: the free-tier API is capped at
125 requests/day, which makes pulling ~5,000 opinions through the paginated
search endpoint impractical. CourtListener's quarterly bulk CSV snapshots
are public on S3, unauthenticated, and have no rate limit — see
TECH_DESIGN.md module 1 for the full rationale.

Run in order:
    python -m ingest.bulk_ingest --check-volume
        Downloads dockets + opinion-clusters (~7.4GB total, one-time) and
        reports an exact count of opinions matching config.COURT_ID +
        config.DATE_START..DATE_END. This is the volume-check gate — don't
        proceed past it until the count looks right. (config.COURT_ID is
        CourtListener's own court identifier, e.g. "ca9" — if you want to
        double check it against their court list, the tiny courts.csv.bz2
        file is the place, but it isn't needed for this step.)

    python -m ingest.bulk_ingest --fetch-text
        Only after the volume check looks good. Streams the ~54GB
        opinions.csv.bz2 once, keeping only rows whose cluster_id matched
        the scoped set, and writes them to data/processed/opinions_scoped.csv.
        This is the expensive one-time step — run it and let it finish,
        don't re-run casually.

    python -m ingest.bulk_ingest --fetch-citations
        Downloads citations.csv.bz2 (~127MB, small) and filters to edges
        touching the scoped opinion set. Feeds citations/build_graph.py.
"""
import argparse
import bz2
import csv
import re
import sys
from urllib.request import urlopen

import requests
from tqdm import tqdm

import config

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # opinion plain_text fields can be large


def latest_bulk_file(table_name):
    """Finds the most recent dated snapshot for a bulk table via the S3
    ListObjectsV2 API (plain HTTP, no auth needed for a public bucket)."""
    url = f"{config.BULK_S3_BASE}/?list-type=2&prefix=bulk-data/{table_name}-"
    with urlopen(url) as resp:
        xml = resp.read().decode("utf-8")
    keys = re.findall(r"<Key>(bulk-data/" + re.escape(table_name) + r"-\d{4}-\d{2}-\d{2}\.csv\.bz2)</Key>", xml)
    if not keys:
        raise RuntimeError(f"No dated bulk file found for table {table_name!r}")
    return sorted(keys)[-1]  # ISO dates sort correctly as strings


def download_bulk_file(table_name):
    """Downloads a bulk CSV (bz2) to data/raw/ if not already present."""
    key = latest_bulk_file(table_name)
    url = f"{config.BULK_S3_BASE}/{key}"
    dest = config.RAW_DIR / key.split("/")[-1]
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{dest.name} already downloaded, skipping.")
        return dest

    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    print(f"Downloading {key} ({total / 1e9:.2f} GB)...")
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def _csv_rows(bz2_path):
    """Yields rows as dicts.

    CourtListener's bulk exports embed literal backslash-escaped quotes
    (e.g. an XML declaration stored as `\"1.0\"`) inside otherwise
    double-quoted CSV fields. That's not standard CSV escaping (RFC4180
    doubles the quote char, `""`) — Python's default 'excel' dialect has no
    concept of a backslash escape, so it reads a lone `\"` as closing the
    field early, and every row after that point can desync (confirmed
    directly: one real opinion row fragmented into 44 phantom rows under
    the default dialect, with body text spilling into columns like `type`).
    escapechar="\\" fixes it; a line-preprocessing alternative was tried and
    measured no faster (both land around ~600-700 rows/s on this data, which
    is simply what processing rows with several huge duplicate HTML/XML
    fields each costs) — kept this version for its simplicity. Full-file
    scans of opinions.csv take several hours at this rate; that's expected,
    not a regression (an earlier "2 billion rows in 5h23m" reading from
    before this fix was phantom fragmented rows, not a valid comparison).
    """
    with bz2.open(bz2_path, mode="rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, escapechar="\\", doublequote=True):
            row.pop(None, None)
            yield row


def filter_dockets_by_court(court_id):
    path = download_bulk_file("dockets")
    print(f"Scanning dockets for court_id={court_id!r}...")
    matched_ids = set()
    for row in tqdm(_csv_rows(path), unit=" rows"):
        if row.get("court_id") == court_id:
            matched_ids.add(row["id"])
    print(f"Matched {len(matched_ids)} dockets in {court_id}.")
    return matched_ids


def filter_clusters(docket_ids, date_start, date_end):
    path = download_bulk_file("opinion-clusters")
    print(f"Scanning opinion-clusters for docket match + date range "
          f"{date_start}..{date_end}...")
    matches = []
    for row in tqdm(_csv_rows(path), unit=" rows"):
        if row.get("docket_id") not in docket_ids:
            continue
        date_filed = row.get("date_filed", "")
        if not (date_start <= date_filed <= date_end):
            continue
        matches.append(row)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DIR / "opinion_clusters_scoped.csv"
    if matches:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=matches[0].keys(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(matches)
    print(f"Matched {len(matches)} opinion clusters. Saved to {out_path}")
    return matches


def check_volume():
    docket_ids = filter_dockets_by_court(config.COURT_ID)
    if not docket_ids:
        print(f"No dockets found for court_id={config.COURT_ID!r} — check "
              f"the court identifier against the 'courts' bulk table.")
        return
    clusters = filter_clusters(docket_ids, config.DATE_START, config.DATE_END)
    print(f"\nTotal opinion clusters in scope: {len(clusters)}")
    print(f"Target: {config.TARGET_OPINION_COUNT}")
    print("Note: this counts ALL opinions in the court/date range — doctrine "
          "keyword filtering (e.g. 'qualified immunity') happens later, once "
          "full text is available, and will narrow this further.")
    if len(clusters) < config.TARGET_OPINION_COUNT:
        print("Below target already at the court/date level — widen the "
              "date range before doctrine filtering narrows it further.")


def _matches_doctrine_strict(plain_text):
    """Original doctrine gate: "qualified immunity" plus 4th-Amendment/
    excessive-force context. On a full run this matched only 839 opinions,
    well under the 5,000 target (see opinions_scoped.csv). Kept as-is —
    its output file is not touched by the loose mode below."""
    text = plain_text.lower()
    if "qualified immunity" not in text:
        return False
    return "excessive force" in text or "fourth amendment" in text


def _matches_doctrine_loose(plain_text):
    """Loosened gate: "qualified immunity" alone, no excessive-force/4th
    Amendment requirement. A 5.5min/~3%-of-file sample check estimated
    ~1,500-2,000 real matches (extrapolation is noisy — the strict version
    of this same estimate overshot the true 839 by ~60%, so treat this as
    an order-of-magnitude figure, not exact). Still short of the original
    5,000 target; accepted deliberately per project decision — solo-project
    scope, not worth widening court/date range to chase the original number."""
    return "qualified immunity" in plain_text.lower()


DOCTRINE_MODES = {
    "strict": (_matches_doctrine_strict, "opinions_scoped.csv"),
    "loose": (_matches_doctrine_loose, "opinions_scoped_loose.csv"),
}


def fetch_text_for_scoped_clusters(mode="strict"):
    """The expensive step: one streaming pass over the ~54GB opinions file,
    keeping only rows whose cluster_id was matched by check_volume() AND
    whose text matches the doctrine gate for the given mode.

    mode selects both the filter and the output filename (see
    DOCTRINE_MODES) so a "loose" run never overwrites the "strict" run's
    output, or vice versa."""
    doctrine_fn, out_name = DOCTRINE_MODES[mode]
    scoped_path = config.PROCESSED_DIR / "opinion_clusters_scoped.csv"
    if not scoped_path.exists():
        raise SystemExit("Run --check-volume first to produce "
                          "opinion_clusters_scoped.csv.")
    with open(scoped_path, encoding="utf-8") as f:
        cluster_ids = {row["id"] for row in csv.DictReader(f)}
    print(f"[{mode}] Filtering opinions.csv.bz2 for {len(cluster_ids)} "
          f"court/date-scoped clusters, narrowed further by doctrine "
          f"keywords in the full text (this downloads ~54GB once — let it "
          f"run)...")

    path = download_bulk_file("opinions")
    out_path = config.PROCESSED_DIR / out_name
    matched = 0
    cluster_matched_no_doctrine = 0
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = None
        for row in tqdm(_csv_rows(path), unit=" rows"):
            if row.get("cluster_id") not in cluster_ids:
                continue
            if not doctrine_fn(row.get("plain_text", "")):
                cluster_matched_no_doctrine += 1
                continue
            if writer is None:
                writer = csv.DictWriter(out_f, fieldnames=row.keys(), extrasaction="ignore")
                writer.writeheader()
            writer.writerow(row)
            matched += 1
    print(f"[{mode}] Matched {matched} opinions (doctrine-scoped). "
          f"{cluster_matched_no_doctrine} were in court/date scope but did not "
          f"match the doctrine gate. Saved to {out_path}")


def fetch_citations_for_scoped_opinions(mode="strict"):
    """Pulls citation graph edges (citing_opinion_id/cited_opinion_id/depth)
    for the scoped opinion set produced by the given mode.

    Confirmed by inspecting the actual columns: the file literally named
    "citations-*.csv.bz2" is CourtListener's search_citation table (reporter
    citation strings per cluster, e.g. "530 U.S. 466" — id/volume/reporter/
    page/cluster_id), NOT the graph edge table. The edge table
    (search_opinionscited: id/depth/cited_opinion_id/citing_opinion_id) is
    published under the name "citation-map-*.csv.bz2" instead — verified by
    downloading its header directly. Easy to get backwards from the bulk
    listing alone; don't assume the name matches the table without checking.
    """
    _, opinions_name = DOCTRINE_MODES[mode]
    scoped_path = config.PROCESSED_DIR / opinions_name
    if not scoped_path.exists():
        raise SystemExit(f"Run --fetch-text --doctrine-mode {mode} first to "
                          f"produce {opinions_name}.")
    with open(scoped_path, encoding="utf-8") as f:
        opinion_ids = {row["id"] for row in csv.DictReader(f)}

    out_name = "citations_scoped.csv" if mode == "strict" else "citations_scoped_loose.csv"
    path = download_bulk_file("citation-map")
    out_path = config.PROCESSED_DIR / out_name
    matched = 0
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = None
        for row in tqdm(_csv_rows(path), unit=" rows"):
            if row.get("citing_opinion_id") not in opinion_ids and \
               row.get("cited_opinion_id") not in opinion_ids:
                continue
            if writer is None:
                writer = csv.DictWriter(out_f, fieldnames=row.keys(), extrasaction="ignore")
                writer.writeheader()
            writer.writerow(row)
            matched += 1
    print(f"[{mode}] Matched {matched} citation edges. Saved to {out_path}")


def fetch_citation_strings_for_scoped_opinions(mode="strict"):
    """Pulls reporter citation strings (e.g. "530 U.S. 466") for the clusters
    behind the scoped opinion set — this is CourtListener's search_citation
    table (see the note in fetch_citations_for_scoped_opinions on why it's
    NOT the graph edge table despite the confusing filename). Needed by
    citations/classify_treatment.py to locate where a cited case is actually
    mentioned in a citing opinion's text — the citation-map edge table only
    says "opinion A cites opinion B," never where in A's text or under what
    citation string."""
    _, opinions_name = DOCTRINE_MODES[mode]
    scoped_path = config.PROCESSED_DIR / opinions_name
    if not scoped_path.exists():
        raise SystemExit(f"Run --fetch-text --doctrine-mode {mode} first to "
                          f"produce {opinions_name}.")
    with open(scoped_path, encoding="utf-8") as f:
        cluster_ids = {row["cluster_id"] for row in csv.DictReader(f)}

    out_name = "citation_strings_scoped.csv" if mode == "strict" else "citation_strings_scoped_loose.csv"
    path = download_bulk_file("citations")
    out_path = config.PROCESSED_DIR / out_name
    matched = 0
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = None
        for row in tqdm(_csv_rows(path), unit=" rows"):
            if row.get("cluster_id") not in cluster_ids:
                continue
            if writer is None:
                writer = csv.DictWriter(out_f, fieldnames=row.keys(), extrasaction="ignore")
                writer.writeheader()
            writer.writerow(row)
            matched += 1
    print(f"[{mode}] Matched {matched} citation strings. Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-volume", action="store_true")
    parser.add_argument("--fetch-text", action="store_true")
    parser.add_argument("--fetch-citations", action="store_true")
    parser.add_argument("--fetch-citation-strings", action="store_true")
    parser.add_argument("--doctrine-mode", choices=["strict", "loose"], default="strict",
                         help="strict = original AND filter (opinions_scoped.csv); "
                              "loose = 'qualified immunity' alone (opinions_scoped_loose.csv)")
    args = parser.parse_args()
    if args.check_volume:
        check_volume()
    elif args.fetch_text:
        fetch_text_for_scoped_clusters(mode=args.doctrine_mode)
    elif args.fetch_citations:
        fetch_citations_for_scoped_opinions(mode=args.doctrine_mode)
    elif args.fetch_citation_strings:
        fetch_citation_strings_for_scoped_opinions(mode=args.doctrine_mode)
    else:
        parser.print_help()
