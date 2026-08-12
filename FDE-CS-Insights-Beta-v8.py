# =============================================================================
# Atkins Consulting - Renewal Intelligence
# Renewal-Trajectory Cohort Agent - Interactive Prototype (Streamlit v2)
# =============================================================================
#
# WHAT THIS FILE IS
# -----------------
# A single-file, branded, multi-page version of the row-13 prototype from the
# project design summary. The MODEL LOGIC is unchanged from the batch/v1 app:
# same four-table synthetic data, same Layers 1-3 and the Layer-4 placeholder,
# same per-term modeling and provisional outcome bands. Everything new here is
# presentation: a company identity, real page-to-page navigation, red/amber/
# green risk signalling, hover explanations on columns, a full sortable account
# portfolio, and a single Assumptions page that carries every caveat.
#
# HOW IT IS ORGANISED (read top to bottom)
# ----------------------------------------
#   PART A - Configuration defaults + brand tokens + colour thresholds
#   PART B - The engine: pure functions, no web code (unchanged model logic)
#   PART C - Caching wrapper so navigation is instant
#   PART D - The web pages (branding, navigation, and each screen)
#
# You do NOT need to read PART B to change how it behaves. Use the sidebar.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
import base64, os


# =============================================================================
# PART A - CONFIGURATION DEFAULTS + BRAND + THRESHOLDS
# =============================================================================
# The synthetic-data settings (mirror "SECTION 0" of the batch script). These
# are only starting values; the sidebar changes them live.

DEFAULT_N_ACCOUNTS    = 400            # historical accounts used to learn the pattern
DEFAULT_N_LIVE        = 40             # live accounts to score in the portfolio
FEATURES              = [f"feature_{i+1}" for i in range(8)]
TERM_CHOICES          = [4, 12]        # contract lengths in QUARTERS (12mo, 36mo)
DEFAULT_TERM_PROBS    = [0.55, 0.45]   # 55% short, 45% long
DEFAULT_NOISE_SHARE   = 0.12           # outcome deliberately contradicts usage
DEFAULT_WINNER_SHARE  = 0.70           # winner vs faller usage-shape split
SEGMENTS              = ["SMB", "Mid", "Enterprise"]
DEFAULT_SEGMENT_PROBS = [0.50, 0.35, 0.15]
SEGMENT_COMMIT        = {"SMB": 50_000, "Mid": 200_000, "Enterprise": 800_000}
INDUSTRIES            = ["SaaS", "Fintech", "Healthcare", "Retail", "Logistics", "Media"]

# --- Per-page background photos ---------------------------------------------
# A different muted photo sits behind each page. Each value can be EITHER a local
# file path (recommended for a live demo - it gets embedded, so nothing loads
# from the internet) OR a full https URL. Missing/blank entries fall back to a
# plain wash, so the app still runs before you add any images.
#
# To use your own: make a folder called "images" next to app.py and drop in six
# royalty-free photos (Pexels.com or Unsplash.com) named as below. Landscape
# shots of people collaborating work best. Or paste a URL in place of the path.
PAGE_BACKGROUNDS = {
    "Home":         "images/home.jpg",
    "The Model":    "images/model.jpg",
    "Portfolio":    "images/portfolio.jpg",
    "Account Plan": "images/account.jpg",
    "Data":         "images/data.jpg",
    "Assumptions":  "images/assumptions.jpg",
}
# How strongly the photo is muted. 0.90 = a faint ghost behind the page.
# Lower it (e.g. 0.82) to show more of the photo; raise it (0.95) for subtler.
BG_SCRIM = 0.60

# --- Brand tokens ------------------------------------------------------------
BRAND       = "Atkins Consulting"
BRAND_MARK  = "AC"
TAGLINE     = "Renewal Intelligence for Customer Success and Forward Deployed Engineering"

INK    = "#0E2233"   # deep slate navy - primary
SURFACE= "#F4F6F8"   # cool off-white background
CARD   = "#FFFFFF"
ACCENT = "#0E7C86"   # restrained teal - the one brand accent
LINE   = "#DDE4EA"
MUTE   = "#5A6B78"
GREEN  = "#1E8E5A"   # on track to renew
AMBER  = "#C77D0A"   # needs attention
RED    = "#C0392B"   # at risk

# --- Red / Amber / Green thresholds -----------------------------------------
# Severity is measured in "spreads" (IQRs) outside the winning band. A driver is
# only recorded once it is more than 0.25 spreads off. These two lines decide the
# colour and are deliberately simple and adjustable (explained on Assumptions).
YELLOW_SEV = 0.25    # at/above this and below RED = amber (needs attention)
RED_SEV    = 1.00    # at/above this = red (clearly off target)

RISK_ORDER = {"At risk": 0, "Needs attention": 1, "On track": 2}
RISK_COLOR = {"At risk": RED, "Needs attention": AMBER, "On track": GREEN}
RISK_BADGE = {"At risk": "\U0001F534 At risk",
              "Needs attention": "\U0001F7E1 Needs attention",
              "On track": "\U0001F7E2 On track"}


# Provisional outcome bands (end-of-term consumption vs commitment). OPEN in the
# design summary; set here to move forward, not decided.
def label_outcome(end_ratio):
    if end_ratio >= 1.15:  return "expansion"
    if end_ratio >= 1.00:  return "full_renewal"
    if end_ratio >= 0.90:  return "90_99"
    if end_ratio >= 0.75:  return "under_90"
    return "churn"

WINNER_OUTCOMES = {"expansion", "full_renewal"}
LOSER_OUTCOMES  = {"under_90", "churn"}   # "90_99" is treated as neutral

# Usage metrics tracked each quarter, and whether higher (+1) or lower (-1) is
# better. Direction makes every score read the same way: separability.
METRIC_DIRECTION = {
    "consumption_vs_commit":     +1,
    "features_used":             +1,
    "outcomes_produced":         +1,
    "active_users":              +1,
    "sentiment_score":           +1,
    "champion_present":          +1,
    "cost_per_outcome":          -1,
    "consumption_concentration": -1,
    "support_tickets":           -1,
    "escalations":               -1,
    "exec_touch_recency":        -1,
}

METRIC_LABELS = {
    "consumption_vs_commit":     "Consumption vs commitment",
    "features_used":             "Adoption breadth (features used)",
    "outcomes_produced":         "Outcomes produced",
    "active_users":              "Active users",
    "sentiment_score":           "Sentiment score",
    "champion_present":          "Champion present",
    "cost_per_outcome":          "Cost per outcome",
    "consumption_concentration": "Consumption concentration",
    "support_tickets":           "Support tickets",
    "escalations":               "Escalations",
    "exec_touch_recency":        "Exec-touch recency (days)",
}

# Short, plain reason each metric matters (used in the action plan help text).
METRIC_WHY = {
    "consumption_vs_commit":     "Using less than they bought signals contraction risk.",
    "features_used":             "Narrow adoption is easier for a customer to walk away from.",
    "outcomes_produced":         "Fewer results means less proof of value at renewal.",
    "active_users":              "Few active users means the product is not embedded.",
    "sentiment_score":           "Falling sentiment tends to lead a drop in usage.",
    "champion_present":          "Losing the internal champion is a top churn driver.",
    "cost_per_outcome":          "Paying more per result erodes the value story.",
    "consumption_concentration": "Usage from one team collapses if that team changes.",
    "support_tickets":           "A heavy support load signals friction and frustration.",
    "escalations":               "Escalations mean unresolved, renewal-threatening issues.",
    "exec_touch_recency":        "No recent executive contact weakens the relationship.",
}


# =============================================================================
# PART B - THE ENGINE  (pure functions; no Streamlit code; model unchanged)
# =============================================================================

def make_trajectory(shape, term, rng, faller_end_level=0.65):
    """Build one account's per-quarter usage numbers.

    Fallers peak mid-term then fade, so mid-term their consumption OVERLAPS
    winners. That is exactly why sentiment and concentration separate winners
    from losers earlier than raw consumption does. We bake that in on purpose.
    """
    rows = []
    for q in range(1, term + 1):
        frac = q / term
        if shape == "winner":
            cons = 0.55 + 0.60 * frac + rng.normal(0, 0.05)
            sentiment = np.clip(0.78 + rng.normal(0, 0.06), 0, 1)
            concentration = np.clip(0.38 + rng.normal(0, 0.06), 0, 1)
            features = min(len(FEATURES), int(round(3 + 4 * frac + rng.normal(0, 0.6))))
            champion = 1 if rng.random() > 0.05 else 0
            exec_recency = max(1, rng.normal(20, 8))
            cost_per_outcome = np.clip(1.00 - 0.15 * frac + rng.normal(0, 0.05), 0.4, 2.0)
            tickets = max(0, int(rng.normal(2, 1)))
            escal = 1 if rng.random() < 0.05 else 0
        else:  # faller
            start = 0.50
            peak = max(faller_end_level + 0.10, 0.85)
            if frac <= 0.45:
                cons = start + (peak - start) * (frac / 0.45)
            else:
                cons = peak + (faller_end_level - peak) * ((frac - 0.45) / 0.55)
            cons = cons + rng.normal(0, 0.04)
            sentiment = np.clip(0.62 - 0.25 * frac + rng.normal(0, 0.07), 0, 1)
            concentration = np.clip(0.58 + 0.15 * frac + rng.normal(0, 0.07), 0, 1)
            features = min(len(FEATURES), int(round(2 + 2 * frac + rng.normal(0, 0.6))))
            champion = 0 if (frac > 0.5 and rng.random() < 0.5) else (1 if rng.random() > 0.2 else 0)
            exec_recency = max(1, rng.normal(55, 18) + 30 * frac)
            cost_per_outcome = np.clip(1.05 + 0.35 * frac + rng.normal(0, 0.07), 0.4, 3.0)
            tickets = max(0, int(rng.normal(5, 2)))
            escal = 1 if rng.random() < 0.30 else 0

        cons = max(0.05, cons)
        features = max(1, features)
        outcomes = max(0.1, cons * (1.0 + 0.1 * features) + rng.normal(0, 0.05))
        active_users = max(1, int(features * rng.normal(6, 1.5)))

        rows.append({
            "quarter_within_term":      q,
            "consumption_vs_commit":    round(cons, 4),
            "features_used":            features,
            "outcomes_produced":        round(outcomes, 4),
            "active_users":             active_users,
            "sentiment_score":          round(sentiment, 4),
            "champion_present":         champion,
            "cost_per_outcome":         round(cost_per_outcome, 4),
            "consumption_concentration":round(concentration, 4),
            "support_tickets":          tickets,
            "escalations":              escal,
            "exec_touch_recency":       round(exec_recency, 1),
        })
    return rows


def generate_dataset(n_accounts, term_probs, segment_probs, noise_share,
                     winner_share, seed):
    """Create the four tables: accounts, account_quarter,
    account_quarter_feature, and a small event_log sample.
    """
    rng = np.random.default_rng(seed)
    acct_rows, aq_rows, feat_rows, event_rows = [], [], [], []

    for i in range(n_accounts):
        aid = f"ACC{i:04d}"
        segment = rng.choice(SEGMENTS, p=segment_probs)
        industry = rng.choice(INDUSTRIES)
        term = int(rng.choice(TERM_CHOICES, p=term_probs))
        start_quarter = f"Q{rng.integers(1,5)}-{rng.integers(2019,2024)}"

        base = SEGMENT_COMMIT[segment]
        committed = int(base * np.exp(rng.normal(0, 0.25)))

        shape = "winner" if rng.random() < winner_share else "faller"
        is_noise = rng.random() < noise_share

        faller_end_level = rng.uniform(0.45, 0.98)
        traj = make_trajectory(shape, term, rng, faller_end_level)
        end_ratio = traj[-1]["consumption_vs_commit"]
        outcome = label_outcome(end_ratio)

        if is_noise:
            if shape == "winner":
                outcome = "churn"
                reason = "budget_cut_or_acquisition"
            else:
                outcome = "full_renewal"
                reason = "switching_cost_retention"
        else:
            reason = {
                "expansion":    "expanded_usage",
                "full_renewal": "healthy_renewal",
                "90_99":        "soft_renewal",
                "under_90":     "under_target",
                "churn":        "did_not_renew",
            }[outcome]

        acct_rows.append({
            "account_id":                 aid,
            "segment":                    segment,
            "industry":                   industry,
            "start_quarter":              start_quarter,
            "term_quarters":              term,
            "committed_credits_quarter":  committed,
            "renewal_quarter":            term,
            "outcome":                    outcome,
            "outcome_reason":             reason,
            "shape_internal":             shape,
            "is_noise_internal":          int(is_noise),
        })

        for r in traj:
            credits_consumed = int(committed * r["consumption_vs_commit"])
            aq_rows.append({
                "account_id":                aid,
                "term_quarters":             term,
                **r,
                "committed_credits":         committed,
                "credits_consumed":          credits_consumed,
            })
            conc = r["consumption_concentration"]
            weights = rng.random(len(FEATURES)) ** (1 + 4 * conc)
            weights = weights / weights.sum()
            for f, w in zip(FEATURES, weights):
                feat_rows.append({
                    "account_id":       aid,
                    "quarter_within_term": r["quarter_within_term"],
                    "feature":          f,
                    "credits_consumed": int(credits_consumed * w),
                    "outcomes_produced":round(r["outcomes_produced"] * w, 4),
                })

        if i < 3:
            for r in traj:
                for _ in range(40):
                    feat = rng.choice(FEATURES)
                    event_rows.append({
                        "account_id":     aid,
                        "user_id":        f"U{rng.integers(1, 50):03d}",
                        "quarter_within_term": r["quarter_within_term"],
                        "feature":        feat,
                        "event_type":     rng.choice(["run", "query", "export", "config"]),
                        "credits":        int(max(1, rng.normal(30, 10))),
                        "produced_outcome": int(rng.random() < 0.6),
                    })

    return {
        "accounts":                pd.DataFrame(acct_rows),
        "account_quarter":         pd.DataFrame(aq_rows),
        "account_quarter_feature": pd.DataFrame(feat_rows),
        "event_log":               pd.DataFrame(event_rows),
    }


def cohort_bands(aq, accounts, term, metric):
    """Layer 1. Median and middle-half band (25th-75th pct) at each
    quarter-within-term, for winners and losers separately.
    """
    outc = accounts.set_index("account_id")["outcome"]
    df = aq[aq["term_quarters"] == term].copy()
    df["group"] = df["account_id"].map(
        lambda a: "winning" if outc[a] in WINNER_OUTCOMES
        else ("losing" if outc[a] in LOSER_OUTCOMES else "neutral")
    )
    out = {}
    for grp in ("winning", "losing"):
        g = df[df["group"] == grp]
        stats = g.groupby("quarter_within_term")[metric].agg(
            median="median",
            lo=lambda s: s.quantile(0.25),
            hi=lambda s: s.quantile(0.75),
        ).reset_index()
        out[grp] = stats
    return out


def auc_score(x, y):
    """Rank-based AUC (Mann-Whitney). 0.5 = no signal, 1.0 = perfect."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    n1, n0 = int(y.sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(x).rank().to_numpy()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def indicator_auc(aq, accounts, term):
    """Layer 2. For each metric at each quarter, how well it separates
    winners from losers.
    """
    outc = accounts.set_index("account_id")["outcome"]
    df = aq[aq["term_quarters"] == term].copy()
    df["label"] = df["account_id"].map(
        lambda a: 1 if outc[a] in WINNER_OUTCOMES else (0 if outc[a] in LOSER_OUTCOMES else -1)
    )
    df = df[df["label"] >= 0]
    rows = []
    for q in sorted(df["quarter_within_term"].unique()):
        sub = df[df["quarter_within_term"] == q]
        for metric, direction in METRIC_DIRECTION.items():
            raw = auc_score(sub[metric] * direction, sub["label"])
            rows.append({"quarter_within_term": q, "metric": metric,
                         "auc": round(raw, 3) if pd.notna(raw) else np.nan})
    return pd.DataFrame(rows)


def severity_status(sev):
    """Turn a gap size (in spreads) into a red / amber / green word."""
    if sev >= RED_SEV:    return "At risk"
    if sev >= YELLOW_SEV: return "Needs attention"
    return "On track"


def account_risk_band(drivers):
    """Roll a list of off-target drivers up to one account-level colour.

    Rule (simple and adjustable): any clearly-off (red) driver makes the account
    At risk; otherwise one or more amber drivers make it Needs attention;
    nothing flagged is On track.
    """
    reds = sum(1 for d in drivers if d["severity"] >= RED_SEV)
    ambers = sum(1 for d in drivers if YELLOW_SEV <= d["severity"] < RED_SEV)
    if reds >= 1:
        return "At risk"
    if ambers >= 1:
        return "Needs attention"
    return "On track"


def driver_read(value, benchmark, direction):
    """Plain-language 'current vs winning' so 1.00 vs 0.94 is never ambiguous."""
    if direction > 0:
        tail = "below target" if value < benchmark else "at/above target"
    else:
        tail = "above target" if value > benchmark else "at/below target"
    return f"{value:.2f} vs winning {benchmark:.2f} ({tail})"


def score_live_accounts(dataset, term_probs, segment_probs, noise_share,
                        winner_share, seed, n_live):
    """Layer 3. Generate a fresh pool of live accounts observed partway through
    their term, hide the true outcome, and score each against the winning band.
    Returns (worklist_df, plans_dict).
    """
    aq_hist = dataset["account_quarter"]
    accts_hist = dataset["accounts"]
    outc = accts_hist.set_index("account_id")["outcome"]
    hist = aq_hist.copy()
    hist["is_winner"] = hist["account_id"].map(lambda a: outc[a] in WINNER_OUTCOMES)
    win = hist[hist["is_winner"]]

    band = {}
    for term in TERM_CHOICES:
        for q in range(1, term + 1):
            sub = win[(win["term_quarters"] == term) & (win["quarter_within_term"] == q)]
            if sub.empty:
                continue
            for metric in METRIC_DIRECTION:
                s = sub[metric]
                band[(term, q, metric)] = (s.quantile(0.25), s.median(), s.quantile(0.75))

    live = generate_dataset(n_live, term_probs, segment_probs, noise_share,
                            winner_share, seed + 999)
    live_acct = live["accounts"]
    live_aq = live["account_quarter"]

    rng = np.random.default_rng(seed + 7)
    work_rows, plans = [], {}

    for _, a in live_acct.iterrows():
        term = a["term_quarters"]
        current_q = max(1, int(round(term * rng.uniform(0.40, 0.80))))
        q_to_renewal = term - current_q

        cur = live_aq[(live_aq["account_id"] == a["account_id"]) &
                      (live_aq["quarter_within_term"] == current_q)]
        if cur.empty:
            continue
        cur = cur.iloc[0]

        drivers = []
        total_gap = 0.0
        n_tracked = 0
        for metric, direction in METRIC_DIRECTION.items():
            key = (term, current_q, metric)
            if key not in band:
                continue
            n_tracked += 1
            q25, med, q75 = band[key]
            iqr = max(q75 - q25, 1e-6)
            val = float(cur[metric])
            if direction > 0:
                sev = max(0.0, (q25 - val) / iqr)
            else:
                sev = max(0.0, (val - q75) / iqr)
            sev = min(sev, 3.0)
            if sev > YELLOW_SEV:
                drivers.append({
                    "metric":    metric,
                    "severity":  round(sev, 2),
                    "value":     val,
                    "benchmark": float(med),
                    "direction": direction,
                    "status":    severity_status(sev),
                    "detail":    driver_read(val, float(med), direction),
                })
                total_gap += sev

        drivers = sorted(drivers, key=lambda d: -d["severity"])
        risk_band = account_risk_band(drivers)

        contract_value = int(a["committed_credits_quarter"]) * term
        value_weight = np.log10(contract_value) / 5.0
        urgency = 1.0 + (1.0 - q_to_renewal / term)
        priority = round(total_gap * value_weight * urgency, 2)

        work_rows.append({
            "account_id":                a["account_id"],
            "segment":                   a["segment"],
            "risk_band":                 risk_band,
            "term_label":                "12-month" if term == 4 else "36-month",
            "term_quarters":             term,
            "committed_credits_quarter": int(a["committed_credits_quarter"]),
            "contract_value":            contract_value,
            "current_quarter":           current_q,
            "quarters_to_renewal":       q_to_renewal,
            "off_target_count":          len(drivers),
            "top_drivers":               ", ".join(METRIC_LABELS[d["metric"]] for d in drivers[:2]) or "None",
            "total_gap":                 round(total_gap, 2),
            "priority_score":            priority,
            "true_outcome_hidden":       a["outcome"],
        })
        plans[a["account_id"]] = {"meta": a.to_dict(), "current_q": current_q,
                                  "q_to_renewal": q_to_renewal, "drivers": drivers,
                                  "risk_band": risk_band, "n_tracked": n_tracked}

    worklist = pd.DataFrame(work_rows)
    if not worklist.empty:
        worklist = worklist.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return worklist, plans


# Owner + metric-to-move mapping (design summary 4.4 / Section 5).
DRIVER_TO_ACTION = {
    "features_used":             ("CSM + FDE",     "Core features adopted"),
    "consumption_vs_commit":     ("FDE + CSM",     "Consumption vs commitment %"),
    "consumption_concentration": ("FDE + CSM",     "Consumption concentration"),
    "cost_per_outcome":          ("FDE + CSM",     "Cost per outcome"),
    "champion_present":          ("CSM",           "Engaged stakeholders"),
    "exec_touch_recency":        ("CSM",           "Exec-touch recency"),
    "sentiment_score":           ("CSM",           "CSAT / sentiment trend"),
    "support_tickets":           ("FDE + support", "Open ticket / severe count"),
    "escalations":               ("FDE + support", "Escalation rate"),
    "outcomes_produced":         ("FDE + CSM",     "Outcomes produced"),
    "active_users":              ("CSM",           "Active-user ratio"),
}


def build_action_plan(account_id, plan):
    """Layer 4 (placeholder). One row per off-target driver: colour, owner, the
    number to move, and current-vs-winning in plain words. 'Recommended play' is
    intentionally blank - that slot is filled by your team's play library.
    """
    steps = []
    for d in plan["drivers"]:
        owner, metric_to_move = DRIVER_TO_ACTION.get(d["metric"], ("CSM", d["metric"]))
        steps.append({
            "Status":             RISK_BADGE[d["status"]],
            "Risk area":          METRIC_LABELS.get(d["metric"], d["metric"]),
            "Why it matters":     METRIC_WHY.get(d["metric"], ""),
            "Owner":              owner,
            "Metric to move":     metric_to_move,
            "Current vs winning": d["detail"],
            "Recommended play":   "",   # <-- play library goes here
        })
    return {
        "account_id":          account_id,
        "segment":             plan["meta"]["segment"],
        "term_quarters":       plan["meta"]["term_quarters"],
        "current_quarter":     plan["current_q"],
        "quarters_to_renewal": plan["q_to_renewal"],
        "risk_band":           plan["risk_band"],
        "on_track_count":      plan["n_tracked"] - len(plan["drivers"]),
        "tracked_count":       plan["n_tracked"],
        "off_target_drivers":  steps,
    }


# =============================================================================
# PART C - CACHING  (so clicking around the app is instant)
# =============================================================================

@st.cache_data(show_spinner=False)
def run_pipeline(n_accounts, short_share, smb, mid, noise_share, winner_share,
                 seed, n_live):
    term_probs = [short_share, 1 - short_share]
    ent = max(0.0, 1 - smb - mid)
    seg = [smb, mid, ent]
    total = sum(seg) or 1.0
    segment_probs = [p / total for p in seg]
    dataset = generate_dataset(n_accounts, term_probs, segment_probs,
                               noise_share, winner_share, seed)
    worklist, plans = score_live_accounts(dataset, term_probs, segment_probs,
                                           noise_share, winner_share, seed, n_live)
    return dataset, worklist, plans


# =============================================================================
# PART D - THE WEB PAGES
# =============================================================================

NAV = ["Home", "The Model", "Portfolio", "Account Plan", "Data", "Assumptions"]


def go(page, account=None):
    """Navigate. Used as an on_click callback so a single click is enough."""
    st.session_state.page = page
    st.session_state.portfolio_focus = None   # leave any Portfolio drill-down
    if account is not None:
        st.session_state.selected_account = account


def _bg_source(ref):
    """Turn a page-background reference into a  url() value.

    Accepts a full http(s) URL (used directly) or a local file path (read and
    embedded as base64, so the published app needs no external image host).
    Returns None if the reference is blank or the file is missing, which tells
    the caller to fall back to a plain wash.
    """
    if not ref:
        return None
    if ref.startswith("http://") or ref.startswith("https://"):
        return f'url("{ref}")'
    if os.path.exists(ref):
        ext = os.path.splitext(ref)[1].lstrip(".").lower() or "jpeg"
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        with open(ref, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f'url("data:image/{mime};base64,{b64}")'
    return None


def inject_page_background(page):
    """Paint a muted, fixed, full-page photo behind the current page.

    A near-opaque scrim (the page's own base colour at BG_SCRIM alpha) sits ON
    TOP of the photo to mute it, and a faint dot grid adds texture. Cards, the
    hero, tables, and the sidebar are all solid white, so they stay in the
    foreground and text stays easy to read. If no image is available for the
    page, a plain wash is used instead.
    """
    src = _bg_source(PAGE_BACKGROUNDS.get(page))
    scrim = f"rgba(237,242,245,{BG_SCRIM})"
    dots = "radial-gradient(rgba(14,34,51,0.05) 1px, transparent 1.4px)"
    if src:
        image = f"{dots}, linear-gradient({scrim}, {scrim}), {src}"
        size  = "24px 24px, cover, cover"
        pos   = "0 0, center, center"
        rep   = "repeat, no-repeat, no-repeat"
    else:
        image = f"{dots}, linear-gradient(180deg, #F7F9FB 0%, #E9F0F4 100%)"
        size  = "24px 24px, auto"
        pos   = "0 0, 0 0"
        rep   = "repeat, no-repeat"
    st.markdown(f"""
    <style>
      .stApp {{
          background-color: #EDF2F5;
          background-image: {image};
          background-size: {size};
          background-position: {pos};
          background-repeat: {rep};
          background-attachment: fixed;
      }}
    </style>
    """, unsafe_allow_html=True)


def inject_css():
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

      .stApp {{ background-color: #EDF2F5; }}
      html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {INK}; }}
      h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif; color: {INK}; letter-spacing: -0.01em; }}
      .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1180px; }}

      /* Brand header */
      .ac-brandbar {{ display:flex; align-items:center; gap:14px; margin: 0 0 4px 0; }}
      .ac-mark {{ width:40px; height:40px; border-radius:10px; background:{INK};
                  color:#fff; font-family:'Space Grotesk'; font-weight:700; font-size:18px;
                  display:flex; align-items:center; justify-content:center; letter-spacing:0.02em; }}
      .ac-word {{ font-family:'Space Grotesk'; font-weight:700; font-size:20px; color:{INK}; line-height:1; }}
      .ac-tag  {{ font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.04em;
                  text-transform:uppercase; color:{MUTE}; margin-top:3px; }}
      .ac-rule {{ height:3px; width:100%; background:linear-gradient(90deg,{INK} 0%,{ACCENT} 55%,transparent 100%);
                  border-radius:2px; margin:8px 0 14px 0; }}

      .ac-eyebrow {{ font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.08em;
                     text-transform:uppercase; color:{ACCENT}; margin-bottom:6px; }}

      /* Nav buttons: render as a flat segmented bar */
      div[data-testid="stHorizontalBlock"] .stButton button {{
          border-radius:8px; border:1px solid {LINE}; background:{CARD}; color:{INK};
          font-weight:600; font-size:0.9rem; padding:0.45rem 0.4rem; }}
      div[data-testid="stHorizontalBlock"] .stButton button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
      .stButton button[kind="primary"], .stButton button[data-testid="baseButton-primary"] {{
          background:{INK}; border:1px solid {INK}; color:#fff; }}
      .stButton button[kind="primary"]:hover {{ background:{ACCENT}; border-color:{ACCENT}; color:#fff; }}

      /* Cell (filter) buttons: compact, narrower, text left-aligned and indented.
         Scoped by the st-key-cellbtn_ class Streamlit adds from the widget key. */
      [class*="st-key-cellbtn_"] button {{
          justify-content:flex-start; text-align:left;
          font-size:0.78rem; line-height:1.05;
          min-height:0; height:auto;
          padding:0.14rem 0.4rem 0.14rem 1.8ch;
          max-width:250px; }}

      /* Hero */
      .ac-hero {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px;
                  padding:34px 36px; box-shadow:0 1px 2px rgba(14,34,51,0.04); }}
      .ac-hero h1 {{ font-size:2.15rem; margin:0 0 10px 0; line-height:1.12; }}
      .ac-hero p  {{ font-size:1.02rem; color:{MUTE}; max-width:640px; margin:0; line-height:1.55; }}

      /* Cards */
      .ac-card {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:18px 20px;
                  height:100%; box-shadow:0 1px 2px rgba(14,34,51,0.04); }}
      .ac-card .step {{ font-family:'IBM Plex Mono', monospace; font-size:12px; color:{ACCENT}; }}
      .ac-card h4 {{ margin:6px 0 6px 0; font-size:1.02rem; }}
      .ac-card p  {{ margin:0; color:{MUTE}; font-size:0.9rem; line-height:1.5; }}

      /* Legend pills */
      .ac-legend {{ display:flex; gap:10px; flex-wrap:wrap; margin:2px 0 10px 0; }}
      .ac-pill {{ display:inline-flex; align-items:center; gap:7px; background:{CARD};
                  border:1px solid {LINE}; border-radius:999px; padding:5px 12px; font-size:0.82rem; color:{INK}; }}
      .ac-dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}

      .ac-note {{ background:#EEF4F5; border-left:3px solid {ACCENT}; border-radius:8px;
                  padding:12px 16px; color:{INK}; font-size:0.9rem; line-height:1.5; }}

      [data-testid="stMetric"] {{ background:{CARD}; border:1px solid {LINE}; border-radius:12px;
                                   padding:12px 16px; }}
      section[data-testid="stSidebar"] {{ background:{CARD}; border-right:1px solid {LINE}; }}

      a, a:visited {{ color:{ACCENT}; }}
      :focus-visible {{ outline:2px solid {ACCENT}; outline-offset:2px; }}
      @media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}

      /* hide Streamlit's own header, toolbar, menu and footer */
      header[data-testid="stHeader"] {{display:none;}}
      [data-testid="stToolbar"] {{display:none;}}
      footer {{visibility:hidden;}}
      #MainMenu {{visibility:hidden;}}
    </style>
    """, unsafe_allow_html=True)


def render_header_and_nav():
    st.markdown(f"""
      <div class="ac-brandbar">
        <div class="ac-mark">{BRAND_MARK}</div>
        <div>
          <div class="ac-word">{BRAND}</div>
          <div class="ac-tag">{TAGLINE}</div>
        </div>
      </div>
      <div class="ac-rule"></div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(NAV))
    for i, name in enumerate(NAV):
        active = st.session_state.page == name
        cols[i].button(name, key=f"nav_{name}", use_container_width=True,
                       type="primary" if active else "secondary",
                       on_click=go, args=(name,))
    st.write("")


def risk_legend():
    st.markdown(f"""
      <div class="ac-legend">
        <span class="ac-pill"><span class="ac-dot" style="background:{RED}"></span> At risk - act now</span>
        <span class="ac-pill"><span class="ac-dot" style="background:{AMBER}"></span> Needs attention</span>
        <span class="ac-pill"><span class="ac-dot" style="background:{GREEN}"></span> On track to renew</span>
      </div>
    """, unsafe_allow_html=True)


def brand_font():
    """Use a clean matplotlib font; fall back silently if unavailable."""
    for fam in ["DejaVu Sans"]:
        try:
            fm.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            pass


# ---------- SIDEBAR ----------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(f"<div class='ac-eyebrow'>Scenario controls</div>", unsafe_allow_html=True)
        st.caption("This prototype runs on synthetic data. Move a control and "
                   "every page recomputes.")

        n_accounts = st.slider("Historical accounts (used to learn the pattern)",
                               100, 800, DEFAULT_N_ACCOUNTS, step=50)
        n_live = st.slider("Live accounts to score (the portfolio)",
                           10, 80, DEFAULT_N_LIVE, step=5)
        winner_share = st.slider("Healthy-account share", 0.40, 0.95,
                                 DEFAULT_WINNER_SHARE, step=0.05,
                                 help="Higher means fewer churners, a healthier book.")
        noise_share = st.slider("Noise share (outcome contradicts usage)",
                                0.0, 0.30, DEFAULT_NOISE_SHARE, step=0.02,
                                help="Accounts built to fool the model.")

        st.markdown("**Contract term mix**")
        short_share = st.slider("Share on 12-month deals", 0.0, 1.0,
                                DEFAULT_TERM_PROBS[0], step=0.05)

        st.markdown("**Segment mix**")
        smb = st.slider("SMB share", 0.0, 1.0, DEFAULT_SEGMENT_PROBS[0], step=0.05)
        mid = st.slider("Mid-market share", 0.0, max(0.0, 1.0 - smb),
                        min(DEFAULT_SEGMENT_PROBS[1], 1.0 - smb), step=0.05)
        ent = round(max(0.0, 1.0 - smb - mid), 2)
        st.caption(f"Enterprise share = {ent:.2f} (fills the remainder)")

        seed = st.number_input("Random seed", 0, 9999, 42, step=1,
                               help="Same seed reproduces the same data.")

        st.markdown("<hr style='border:none;border-top:1px solid %s;margin:14px 0'>" % LINE,
                    unsafe_allow_html=True)
        st.button("What are the assumptions?", use_container_width=True,
                  on_click=go, args=("Assumptions",))
        st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:10px;"
                    f"color:{MUTE};margin-top:10px'>{BRAND} - exploration prototype</div>",
                    unsafe_allow_html=True)

    return dict(n_accounts=n_accounts, n_live=n_live, winner_share=winner_share,
                noise_share=noise_share, short_share=short_share, smb=smb, mid=mid,
                seed=seed)


# ---------- HOME -------------------------------------------------------------
def page_home(worklist):
    at_risk = int((worklist["risk_band"] == "At risk").sum()) if not worklist.empty else 0
    soon = int(((worklist["risk_band"] == "At risk") &
                (worklist["quarters_to_renewal"] <= 2)).sum()) if not worklist.empty else 0

    st.markdown(f"""
      <div class="ac-hero">
        <h1>See which customers are drifting off the renewal path,<br>a quarter before the renewal conversation.</h1>
        <p>Atkins Consulting learns what a winning account looks like at each quarter of its
        contract, places every live account on that curve, and turns the gap into a ranked,
        owner-assigned action plan. Built for customer success and forward deployed engineering teams.</p>
      </div>
    """, unsafe_allow_html=True)
    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Live accounts scored", 0 if worklist.empty else len(worklist))
    c2.metric("Flagged at risk", at_risk)
    c3.metric("At risk and renewing soon", soon,
              help="At risk with two or fewer quarters to renewal.")
    st.write("")

    st.markdown("<div class='ac-eyebrow'>How it works</div>", unsafe_allow_html=True)
    cards = [
        ("01", "Learn the winning shape",
         "From history, plot how usage evolved for accounts that renewed or expanded versus those that fell short. That is the benchmark band."),
        ("02", "Find the early signals",
         "Measure which usage signals actually separated winners from losers, and at which quarter. Earlier and sharper is more useful."),
        ("03", "Place every live account",
         "Score each current account against the winning band at its own quarter, and rank the book by risk, size, and time to renewal."),
        ("04", "Hand over an action plan",
         "For each off-track account, produce a plan: the areas behind target, who owns each, and the number to move before renewal."),
    ]
    cc = st.columns(4)
    for col, (step, title, body) in zip(cc, cards):
        col.markdown(f"""<div class="ac-card"><div class="step">{step}</div>
                     <h4>{title}</h4><p>{body}</p></div>""", unsafe_allow_html=True)
    st.write("")

    b1, b2, b3 = st.columns(3)
    b1.button("Open the account portfolio", use_container_width=True, type="primary",
              on_click=go, args=("Portfolio",))
    b2.button("See how the model learns", use_container_width=True,
              on_click=go, args=("The Model",))
    b3.button("Read the assumptions", use_container_width=True,
              on_click=go, args=("Assumptions",))


# ---------- THE MODEL --------------------------------------------------------
def page_model(dataset):
    accounts, aq = dataset["accounts"], dataset["account_quarter"]
    st.header("How the model learns the renewal path")

    tab1, tab2 = st.tabs(["Winning vs losing path", "Which signals predict, and when"])

    with tab1:
        st.write("Pick a contract length and a usage metric. The green band is the "
                 "range for accounts that renewed or expanded; the red band is the "
                 "range for accounts that fell short. The moment they separate is the "
                 "moment this metric can tell a winner from a loser.")
        col_a, col_b = st.columns(2)
        term_sel = col_a.selectbox("Contract term", TERM_CHOICES,
                                   format_func=lambda t: "12-month (4 quarters)" if t == 4
                                   else "36-month (12 quarters)")
        metric_sel = col_b.selectbox("Usage metric", list(METRIC_LABELS.keys()),
                                     format_func=lambda m: METRIC_LABELS[m])
        bands = cohort_bands(aq, accounts, term_sel, metric_sel)
        fig, ax = plt.subplots(figsize=(9, 4.1))
        for grp, color, lbl in (("winning", GREEN, "Renewed / expanded"),
                                ("losing", RED, "Fell short")):
            s = bands[grp]
            if s.empty:
                continue
            ax.plot(s["quarter_within_term"], s["median"], color=color, linewidth=2.4, label=lbl)
            ax.fill_between(s["quarter_within_term"], s["lo"], s["hi"], color=color, alpha=0.13)
        ax.set_xlabel("Quarter within contract term")
        ax.set_ylabel(METRIC_LABELS[metric_sel])
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        st.pyplot(fig)

    with tab2:
        st.write("Each cell scores one signal at one quarter. Read it as separation "
                 "power: 0.50 means the signal tells you nothing yet, 1.00 means it "
                 "perfectly splits winners from losers. Greener and further left is "
                 "better, because it is a clearer signal that arrives earlier.")
        term_auc = st.selectbox("Contract term ", TERM_CHOICES,
                                format_func=lambda t: "12-month (4 quarters)" if t == 4
                                else "36-month (12 quarters)", key="auc_term")
        auc_df = indicator_auc(aq, accounts, term_auc)
        pivot = auc_df.pivot(index="metric", columns="quarter_within_term", values="auc")
        pivot.index = [METRIC_LABELS.get(m, m) for m in pivot.index]
        pivot.index.name = "Signal"
        pivot.columns = [f"Q{c}" for c in pivot.columns]
        st.dataframe(pivot.style.background_gradient(cmap="Greens", vmin=0.5, vmax=1.0)
                     .format("{:.2f}"), use_container_width=True)
        st.markdown("<div class='ac-note'>In this synthetic data, sentiment and "
                    "consumption concentration separate winners from losers earlier "
                    "than raw consumption, because a single consumption reading lags "
                    "the trend. To be re-checked on real data.</div>", unsafe_allow_html=True)


# ---------- PORTFOLIO --------------------------------------------------------
SOON_MAX_QTRS = 2   # "renewing soon" = this many quarters or fewer to renewal
ZONE_BASE     = {"At risk": 0, "Needs attention": 1, "On track": 2}  # chart rows
ZONE_DOT      = {"At risk": "\U0001F534", "Needs attention": "\U0001F7E1",
                 "On track": "\U0001F7E2"}


def _classify_cells(df):
    """Tag each account with its time half: renewing soon vs more time."""
    df = df.copy()
    df["_soon"] = df["quarters_to_renewal"] <= SOON_MAX_QTRS
    return df


def _sort_risk_then_renewal(df):
    """Default order: risk band (red, amber, green), then soonest renewal first."""
    d = df.copy()
    d["_r"] = d["risk_band"].map(RISK_ORDER)
    d = d.sort_values(["_r", "quarters_to_renewal"], ascending=[True, True])
    return d.drop(columns="_r")


def _quadrant_chart(df, big=False):
    """Risk (vertical zones) vs time-to-renewal (horizontal). No account labels.

    Vertical position is driven by the RISK BAND, not the raw gap, so colour and
    height always agree: green in the top zone, amber in the middle, red at the
    bottom, regardless of time to renewal. Within a zone a bigger gap sits a
    little lower; tiny jitter stops bubbles overlapping.
    """
    fig, ax = plt.subplots(figsize=(10.4, 5.2) if big else (7.8, 3.7))
    xmax = max(df["quarters_to_renewal"].max(), 3) + 0.5

    jrng = np.random.default_rng(7)
    d = df.copy()
    d["_xj"] = d["quarters_to_renewal"] + jrng.normal(0, 0.07, len(d))

    def _yval(sub):
        g = sub["total_gap"].to_numpy(dtype=float)
        lo, hi = g.min(), g.max()
        norm = (g - lo) / (hi - lo) if hi > lo else np.full_like(g, 0.5)
        within = 0.15 + 0.70 * (1.0 - norm)            # more gap -> lower in the zone
        base = sub["risk_band"].map(ZONE_BASE).to_numpy(dtype=float)
        return base + within + jrng.normal(0, 0.015, len(sub))

    d["_yj"] = 0.0
    for band in ZONE_BASE:
        m = d["risk_band"] == band
        if m.any():
            d.loc[m, "_yj"] = _yval(d[m])

    ax.axhspan(2, 3, color=GREEN, alpha=0.05)
    ax.axhspan(1, 2, color=AMBER, alpha=0.05)
    ax.axhspan(0, 1, color=RED,   alpha=0.06)
    ax.axvline(SOON_MAX_QTRS + 0.5, color=INK, alpha=0.15, linestyle="--", linewidth=1)

    smax = max(d["contract_value"].max(), 1)
    for band in ["On track", "Needs attention", "At risk"]:
        sub = d[d["risk_band"] == band]
        if sub.empty:
            continue
        sizes = 40 + (sub["contract_value"] / smax) * (460 if big else 320)
        ax.scatter(sub["_xj"], sub["_yj"], s=sizes,
                   c=RISK_COLOR[band], alpha=0.85, edgecolor="white", linewidth=1.2)

    ax.text(0.0, 0.12, "Act now", color=RED, fontsize=10, va="bottom",
            fontfamily="monospace")
    ax.set_xlim(-0.5, xmax)
    ax.set_ylim(0, 3)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["At risk", "Needs attention", "On track"])
    ax.set_xlabel("Time to renewal:  sooner \u2190          \u2192 later")
    ax.grid(axis="x", alpha=0.15)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    return fig


def _portfolio_table(df_sorted, key):
    """Clickable account list; clicking a row opens that account's plan."""
    show = df_sorted.reset_index(drop=True).copy()
    show["risk_band"] = show["risk_band"].map(RISK_BADGE)
    show = show[["account_id", "risk_band", "segment", "term_label",
                 "current_quarter", "quarters_to_renewal", "off_target_count",
                 "top_drivers", "contract_value", "priority_score",
                 "true_outcome_hidden"]]

    st.caption("Click any row to open that account's plan.")
    event = st.dataframe(
        show, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key=key,
        column_config={
            "account_id":          st.column_config.TextColumn("Account", help="Account identifier"),
            "risk_band":           st.column_config.TextColumn("Risk", help="Overall renewal risk"),
            "segment":             st.column_config.TextColumn("Segment", help="Customer size tier"),
            "term_label":          st.column_config.TextColumn("Term", help="Contract length"),
            "current_quarter":     st.column_config.NumberColumn("Now Q", help="Current quarter in term"),
            "quarters_to_renewal": st.column_config.NumberColumn("Qtrs to renewal", help="Quarters until renewal"),
            "off_target_count":    st.column_config.NumberColumn("Off-target", help="Signals behind target"),
            "top_drivers":         st.column_config.TextColumn("Main gaps", help="Largest off-target signals"),
            "contract_value":      st.column_config.NumberColumn("Contract value", help="Credits over full term", format="%d"),
            "priority_score":      st.column_config.NumberColumn("Priority", help="Higher means act sooner", format="%.2f"),
            "true_outcome_hidden": st.column_config.TextColumn("Actual (hidden)", help="Real result, model never sees it"),
        })

    rows = event.selection["rows"]
    if rows:
        picked_id = show.iloc[rows[0]]["account_id"]
        st.session_state.pop(key, None)     # clear selection so it doesn't re-fire
        go("Account Plan", picked_id)
        st.rerun()

    st.caption("The last column is the true outcome, hidden from the model and shown "
               "only so you can sanity-check the ranking by eye.")


def _focus_cell(band, soon):
    """Callback: drill into one chart cell (risk band x time half)."""
    st.session_state.portfolio_focus = (band, bool(soon))
    st.session_state.pop("portfolio_table", None)


def _clear_focus():
    st.session_state.portfolio_focus = None
    st.session_state.pop("portfolio_focus_table", None)


def _cell_label(band, soon):
    return f"{band} \u00b7 {'renewing soon' if soon else 'more time'}"


def page_portfolio(worklist):
    st.header("Account portfolio")

    if worklist.empty:
        st.info("No live accounts to show. Raise 'Live accounts to score' in the sidebar.")
        return

    # A clicked chart cell drills into just those accounts, then stops here.
    focus = st.session_state.get("portfolio_focus")
    if focus:
        _portfolio_focus_view(worklist, focus)
        return

    st.write("Every live account, scored against the winning path at its own quarter. "
             "Use this to find the accounts that are both high risk and close to renewal.")

    # ---- controls ----
    c1, c2 = st.columns(2)
    sort_choice = c1.selectbox("Sort by", [
        "Risk, then soonest renewal", "Soonest renewal first",
        "Highest priority first", "Largest accounts first"])
    bands_pick = c2.multiselect("Show risk levels",
                                ["At risk", "Needs attention", "On track"],
                                default=["At risk", "Needs attention", "On track"])

    df = worklist[worklist["risk_band"].isin(bands_pick)].copy()
    if df.empty:
        st.info("No accounts match the selected risk levels. Add one back in "
                "'Show risk levels' above.")
        return
    df = _classify_cells(df)

    # ---- chart (left) + clickable cells (right) ----
    st.markdown("<div class='ac-eyebrow' style='margin-top:6px'>Risk vs time to renewal</div>",
                unsafe_allow_html=True)
    left, right = st.columns([2, 1])
    with left:
        st.pyplot(_quadrant_chart(df))
    with right:
        st.markdown("<div class='ac-eyebrow'>Open a cell</div>", unsafe_allow_html=True)
        st.caption("Click a cell to see just those accounts, expanded.")
        for band in ["On track", "Needs attention", "At risk"]:
            for soon in (True, False):
                n = int(((df["risk_band"] == band) & (df["_soon"] == soon)).sum())
                when = "soon" if soon else "later"
                slug = band.replace(" ", "").lower()
                st.button(f"{ZONE_DOT[band]} {band} \u00b7 {when} ({n})",
                          key=f"cellbtn_{slug}_{'soon' if soon else 'later'}",
                          use_container_width=True, disabled=(n == 0),
                          on_click=_focus_cell, args=(band, soon))

    st.caption("Vertical is risk, horizontal is time to renewal (sooner on the left). "
               "Bubble size is contract value; the dashed line marks about two quarters "
               "out, so the lower-left cell - at risk and renewing soon - is act-now.")

    # ---- sortable table ----
    if sort_choice == "Risk, then soonest renewal":
        df = _sort_risk_then_renewal(df)
    elif sort_choice == "Soonest renewal first":
        df = df.sort_values(["quarters_to_renewal", "total_gap"], ascending=[True, False])
    elif sort_choice == "Largest accounts first":
        df = df.sort_values("contract_value", ascending=False)
    else:
        df = df.sort_values("priority_score", ascending=False)

    _portfolio_table(df, key="portfolio_table")


def _portfolio_focus_view(worklist, focus):
    """Expanded chart + list for a single clicked cell (risk band x time half)."""
    band, soon = focus
    df = _classify_cells(worklist)
    sub = df[(df["risk_band"] == band) & (df["_soon"] == soon)].copy()

    top = st.columns([3, 1])
    top[0].markdown(f"<div class='ac-eyebrow'>Focused cell</div>"
                    f"<h3 style='margin:2px 0 0 0'>{_cell_label(band, soon)} "
                    f"&middot; {len(sub)} account(s)</h3>", unsafe_allow_html=True)
    top[1].button("\u2190 Back to full portfolio", use_container_width=True,
                  on_click=_clear_focus)

    if sub.empty:
        st.info("No accounts in this cell right now.")
        return

    st.pyplot(_quadrant_chart(sub, big=True))
    st.caption("Only the accounts in this cell are plotted, on the same axes as the "
               "full portfolio so you can see where they sit.")

    _portfolio_table(_sort_risk_then_renewal(sub), key="portfolio_focus_table")


# ---------- ACCOUNT PLAN -----------------------------------------------------
def page_account_plan(worklist, plans):
    st.header("Account action plan")

    if worklist.empty or not plans:
        st.info("No accounts to plan yet. Open the Portfolio tab first.")
        return

    ids = worklist["account_id"].tolist()
    default_id = st.session_state.get("selected_account")
    if default_id not in ids:
        default_id = ids[0]

    c1, c2 = st.columns([2, 1])
    chosen = c1.selectbox("Account", ids, index=ids.index(default_id))
    c2.button("\u2190 Back to portfolio", use_container_width=True,
              on_click=go, args=("Portfolio",))
    st.session_state.selected_account = chosen

    plan = build_action_plan(chosen, plans[chosen])
    risk = plan["risk_band"]

    st.markdown(f"""
      <div class="ac-card" style="border-left:5px solid {RISK_COLOR[risk]}; margin-bottom:14px">
        <div class="step" style="color:{RISK_COLOR[risk]}">{RISK_BADGE[risk]}</div>
        <h4 style="margin-top:4px">{chosen} &middot; {plan['segment']}</h4>
        <p>Quarter {plan['current_quarter']} of {plan['term_quarters']} &middot;
        {plan['quarters_to_renewal']} quarter(s) to renewal &middot;
        {plan['on_track_count']} of {plan['tracked_count']} tracked signals on track.</p>
      </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='ac-note'>This plan lists only the areas where the account "
                "is behind the accounts that renewed. Each row names the team that should "
                "act, the number to improve, and how far the account is from the target "
                "today. The <b>Recommended play</b> column is empty on purpose: that is "
                "where your team's proven step goes (for example, \"run an executive "
                "business review\"). We leave it blank because the right play depends on "
                "your company's own playbook, which a real deployment would load in.</div>",
                unsafe_allow_html=True)
    st.write("")

    if plan["off_target_drivers"]:
        pdf = pd.DataFrame(plan["off_target_drivers"])
        st.dataframe(pdf, use_container_width=True, hide_index=True, column_config={
            "Status":             st.column_config.TextColumn("Status", help="Red, amber or green"),
            "Risk area":          st.column_config.TextColumn("Risk area", help="Signal that is off target"),
            "Why it matters":     st.column_config.TextColumn("Why it matters", help="Plain reason it predicts churn"),
            "Owner":              st.column_config.TextColumn("Owner", help="Team responsible to act"),
            "Metric to move":     st.column_config.TextColumn("Metric to move", help="The number to improve"),
            "Current vs winning": st.column_config.TextColumn("Current vs winning", help="Now versus the winning benchmark"),
            "Recommended play":   st.column_config.TextColumn("Recommended play", help="Your playbook step (blank)"),
        })
    else:
        st.success("No areas off target at this quarter. This account is tracking the "
                   "winning path and is on course to renew.")

    with st.expander("Raw action plan (JSON, as the batch prototype writes it)"):
        st.json(plan)


# ---------- DATA -------------------------------------------------------------
def page_data(dataset):
    st.header("Data")
    st.write("These four tables are worked examples of the exact shapes your real data "
             "must match to run this on live accounts. Download any of them to see the "
             "columns.")
    labels = {
        "accounts": "One row per account: who they are, and how their renewal ended.",
        "account_quarter": "One row per account per quarter: every usage signal we track.",
        "account_quarter_feature": "Usage split by feature, for consumption breakdowns.",
        "event_log": "A sample of raw events for a few accounts, the atomic grain.",
    }
    for name in ["accounts", "account_quarter", "account_quarter_feature", "event_log"]:
        df = dataset[name]
        st.markdown(f"**{name}** &nbsp;<span style='color:{MUTE}'>&mdash; {labels[name]} "
                    f"({len(df):,} rows)</span>", unsafe_allow_html=True)
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
        st.download_button(f"Download {name}.csv",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name=f"{name}.csv", mime="text/csv", key=f"dl_{name}")
        st.write("")


# ---------- ASSUMPTIONS ------------------------------------------------------
def page_assumptions():
    st.header("Assumptions and how to read the results")

    st.markdown("<div class='ac-note'>This is an <b>exploration prototype</b> running on "
                "<b>synthetic (invented) data</b>. Its job is to prove the logic before "
                "real data is gathered, not to make live renewal decisions. The outcome "
                "bands and the way accounts are grouped are provisional choices, set so we "
                "can move forward.</div>", unsafe_allow_html=True)
    st.write("")

    st.subheader("What the colours mean")
    risk_legend()
    st.write("Each account is compared to accounts that renewed, at the same point in "
             "their contract. We measure how far off it is in \"spreads\" (the normal "
             "range of the winning group).")
    st.markdown(
        f"- <b style='color:{GREEN}'>Green, on track</b>: within or better than the winning range.\n"
        f"- <b style='color:{AMBER}'>Amber, needs attention</b>: modestly outside the range on at least one signal.\n"
        f"- <b style='color:{RED}'>Red, at risk</b>: clearly outside the range (more than one full spread) on at least one signal.",
        unsafe_allow_html=True)
    st.write('"Current vs winning" reads as the account\'s number versus the winning '
             'benchmark, with plain words for direction. For example, "0.78 vs winning '
             '0.94 (below target)" means the account is behind; "1.00 vs winning 0.94 '
             '(at/above target)" means it is fine on that signal.')

    st.subheader("Provisional outcome bands")
    st.write("An account's fate is labelled from where its end-of-term consumption lands "
             "against what it committed to buy.")
    st.table(pd.DataFrame({
        "End-of-term consumption vs commitment": ["115% or more", "100% to 115%",
                                                  "90% to 100%", "75% to 90%", "below 75%"],
        "Outcome": ["Expansion", "Full renewal", "90-99% (neutral)", "Under 90%", "Churn"],
    }))

    st.subheader("Why we model 12-month and 36-month contracts separately")
    st.write("A 12-month account has to reach healthy usage far faster than a 36-month "
             "one, so the benchmark bands, scoring, and urgency are all computed per term "
             "length on a quarter-within-term axis. The winning and losing paths separate "
             "at about the same fraction of the term for both, which in real time arrives "
             "roughly three times sooner for a 12-month deal.")

    st.subheader("The noise accounts")
    st.write("About one in eight accounts is deliberate \"noise\": strong usage that still "
             "churns (a budget cut or acquisition), or weak usage that renews anyway (too "
             "costly to switch). They are included so we can later test how often the model "
             "is fooled, rather than assuming it never is.")

    st.subheader("Honest limitations")
    st.markdown(
        "- The synthetic data is cleaner than a real book, so real bands will be noisier.\n"
        "- The score is a first-pass rule, not a trained model, and it depends on which quarter you look at.\n"
        "- Priority is weighted by account value, so a large under-performing account can outrank a small churning one. That is a deliberate triage choice, and it is adjustable.\n"
        "- Only the first contract term is modelled, so survivorship is handled lightly.\n"
        "- The play library (the specific recommended actions) is the one piece only your organisation can supply.")

    st.write("")
    st.button("Back to home", type="primary", on_click=go, args=("Home",))


# =============================================================================
# ROUTER
# =============================================================================
def main():
    st.set_page_config(page_title=f"{BRAND} | Renewal Intelligence",
                       page_icon="\u25C9", layout="wide")
    inject_css()
    brand_font()

    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "selected_account" not in st.session_state:
        st.session_state.selected_account = None
    if "portfolio_focus" not in st.session_state:
        st.session_state.portfolio_focus = None

    settings = render_sidebar()
    dataset, worklist, plans = run_pipeline(
        settings["n_accounts"], settings["short_share"], settings["smb"],
        settings["mid"], settings["noise_share"], settings["winner_share"],
        settings["seed"], settings["n_live"])

    inject_page_background(st.session_state.page)
    render_header_and_nav()

    page = st.session_state.page
    if page == "Home":
        page_home(worklist)
    elif page == "The Model":
        page_model(dataset)
    elif page == "Portfolio":
        page_portfolio(worklist)
    elif page == "Account Plan":
        page_account_plan(worklist, plans)
    elif page == "Data":
        page_data(dataset)
    elif page == "Assumptions":
        page_assumptions()


if __name__ == "__main__":
    main()
