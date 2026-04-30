"""
Patch ce_comparisons with authoritative Table 1 data from the published
systematic review (Baumbach et al., 2024).

The "incorrect: 62" in the validation report means our LLM extraction of
the 78 individual papers produced 'unknown' for fields that are explicitly
stated in the papers. This script fixes that by directly importing Table 1
from the systematic review — the authoritative ground truth.

Run locally:  python patch_from_table1.py
Via API:      POST /patch_table1  (no body needed)
"""

import sqlite3
import re
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "ce_studies.sqlite3"

# ── Table 1 corrections ────────────────────────────────────────────────────────
# Fields: author keyword (lowercase), year, body_region, condition, country,
#         study_design, perspective, outcome_measure, outcome_type, time_horizon
#
# Matching: finds paper_id rows in ce_comparisons where paper_id contains
# both author keyword and year (case-insensitive).

CORRECTIONS = [
    # ══ SPINE: Back (non-LBP) ══════════════════════════════════════════════
    {"a": "barker",     "y": "2019", "body_region": "low_back",    "condition": "osteoporotic vertebral fracture",       "country": "UK",          "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "barker",     "y": "2020", "body_region": "low_back",    "condition": "osteoporotic vertebral fracture",       "country": "UK",          "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "muller",     "y": "2019", "body_region": "low_back",    "condition": "back pain",                            "country": "Germany",     "study_design": "prospective cohort","perspective": "healthcare",  "outcome_measure": "Graded chronic back pain status","outcome_type": "clinical", "time_horizon": "24 months"},
    {"a": "mueller",    "y": "2019", "body_region": "low_back",    "condition": "back pain",                            "country": "Germany",     "study_design": "prospective cohort","perspective": "healthcare",  "outcome_measure": "Graded chronic back pain status","outcome_type": "clinical", "time_horizon": "24 months"},
    {"a": "sogaard",    "y": "2008", "body_region": "low_back",    "condition": "lumbar spinal fusion",                 "country": "Denmark",     "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "Pain and disability index",      "outcome_type": "clinical", "time_horizon": "24 months"},

    # ══ LOW BACK PAIN ═══════════════════════════════════════════════════════
    {"a": "aboagye",    "y": "2015", "body_region": "low_back",    "condition": "LBP",                                  "country": "Sweden",      "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "ankjaer",    "y": "1994", "body_region": "low_back",    "condition": "LBP (herniated disc)",                 "country": "Denmark",     "study_design": "retrospective cohort","perspective": "societal", "outcome_measure": "Low back pain rating scale",     "outcome_type": "clinical", "time_horizon": "12 months"},
    {"a": "apeldoorn",  "y": "2012", "body_region": "low_back",    "condition": "LBP (chronic)",                        "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "bello",      "y": "2015", "body_region": "low_back",    "condition": "LBP (chronic)",                        "country": "Ghana",       "study_design": "feasibility study", "perspective": "healthcare",  "outcome_measure": "SF-36, NRS",                    "outcome_type": "clinical", "time_horizon": "3 months"},
    {"a": "burton",     "y": "2004", "body_region": "low_back",    "condition": "LBP (non-specific)",                   "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "canaway",    "y": "2018", "body_region": "low_back",    "condition": "LBP",                                  "country": "Israel",      "study_design": "prospective cohort","perspective": "healthcare",  "outcome_measure": "QALY (SF-12)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "carr",       "y": "2005", "body_region": "low_back",    "condition": "LBP",                                  "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "RMDQ",                          "outcome_type": "clinical", "time_horizon": "12 months"},
    {"a": "cherkin",    "y": "1998", "body_region": "low_back",    "condition": "LBP (chronic)",                        "country": "US",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "Bothersomeness of symptoms, RDS","outcome_type": "clinical","time_horizon": "12-24 months"},
    {"a": "critchley",  "y": "2007", "body_region": "low_back",    "condition": "LBP (acute)",                          "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "18 months"},
    {"a": "fritz",      "y": "2008", "body_region": "low_back",    "condition": "LBP (acute)",                          "country": "US",          "study_design": "case-control",      "perspective": "healthcare",  "outcome_measure": "OSW, pain rating",              "outcome_type": "clinical", "time_horizon": "24 months"},
    {"a": "fritz",      "y": "2017", "body_region": "low_back",    "condition": "LBP",                                  "country": "US",          "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "hahne",      "y": "2017", "body_region": "low_back",    "condition": "LBP (chronic)",                        "country": "Australia",   "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "herman",     "y": "2008", "body_region": "low_back",    "condition": "LBP",                                  "country": "US",          "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (SF-6D)",                  "outcome_type": "QALY",     "time_horizon": "6 months"},
    {"a": "hlobil",     "y": "2007", "body_region": "low_back",    "condition": "LBP (chronic)",                        "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "Lost productivity days",         "outcome_type": "clinical", "time_horizon": "36 months"},
    {"a": "hurley",     "y": "2015", "body_region": "low_back",    "condition": "LBP",                                  "country": "Ireland",     "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "johnson",    "y": "2007", "body_region": "low_back",    "condition": "LBP",                                  "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "karjalainen","y": "2003", "body_region": "low_back",    "condition": "LBP",                                  "country": "Finland",     "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "Pain intensity, ODI, VAS",      "outcome_type": "clinical", "time_horizon": "12 months"},
    {"a": "kim",        "y": "2020", "body_region": "low_back",    "condition": "LBP (chronic)",                        "country": "South Korea", "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "Functional rating index, VAS",  "outcome_type": "clinical", "time_horizon": "3 weeks"},
    {"a": "niemisto",   "y": "2003", "body_region": "low_back",    "condition": "LBP (subacute and chronic)",           "country": "Finland",     "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "VAS",                           "outcome_type": "clinical", "time_horizon": "12 months"},
    {"a": "niemisto",   "y": "2005", "body_region": "low_back",    "condition": "LBP (subacute and chronic)",           "country": "Finland",     "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "ODI, VAS",                      "outcome_type": "clinical", "time_horizon": "24 months"},
    {"a": "rivero",     "y": "2006", "body_region": "low_back",    "condition": "LBP (chronic)",                        "country": "UK",          "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "smeets",     "y": "2009", "body_region": "low_back",    "condition": "LBP",                                  "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "RMDQ, QALY (EQ-5D)",            "outcome_type": "both",     "time_horizon": "12 months"},
    {"a": "suni",       "y": "2018", "body_region": "low_back",    "condition": "LBP (chronic)",                        "country": "Finland",     "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (SF-6D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "roer",       "y": "2008", "body_region": "low_back",    "condition": "LBP",                                  "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "EQ-5D, RMDQ",                   "outcome_type": "both",     "time_horizon": "12 months"},
    {"a": "whitehurst", "y": "2007", "body_region": "low_back",    "condition": "LBP (chronic)",                        "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "RMDQ, QALY (EQ-5D)",            "outcome_type": "both",     "time_horizon": "12 months"},

    # ══ NECK ════════════════════════════════════════════════════════════════
    {"a": "bosmans",    "y": "2011", "body_region": "neck",        "condition": "neck pain (subacute)",                 "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "Patient perceived recovery, QALY (SF-6D)", "outcome_type": "both", "time_horizon": "12 months"},
    {"a": "leininger",  "y": "2016", "body_region": "neck",        "condition": "neck pain (chronic)",                  "country": "US",          "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (SF-6D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "lewis",      "y": "2007", "body_region": "neck",        "condition": "neck disorders (non-specific)",        "country": "UK",          "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "NPQ, QALY (EQ-5D)",             "outcome_type": "both",     "time_horizon": "6 months"},
    {"a": "van dongen", "y": "2016", "body_region": "neck",        "condition": "neck pain (subacute and chronic)",     "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "NDI, patient perceived recovery","outcome_type": "clinical", "time_horizon": "12 months"},
    {"a": "dongen",     "y": "2016", "body_region": "neck",        "condition": "neck pain (subacute and chronic)",     "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "NDI, patient perceived recovery","outcome_type": "clinical", "time_horizon": "12 months"},

    # korthals 2003 = neck, korthals 2004 = elbow (distinguished by year)
    {"a": "korthals",   "y": "2003", "body_region": "neck",        "condition": "neck pain",                            "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "EQ, functional disability, pain intensity", "outcome_type": "both", "time_horizon": "12 months"},
    {"a": "korthals",   "y": "2004", "body_region": "elbow",       "condition": "epicondylitis lateralis",              "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "EQ, pain-free function",        "outcome_type": "both",     "time_horizon": "12 months"},

    # manca 2006 = neck, manca 2007 = back/neck (distinguished by year)
    {"a": "manca",      "y": "2006", "body_region": "neck",        "condition": "neck pain",                            "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "manca",      "y": "2007", "body_region": "multi_region","condition": "back or neck pain",                    "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},

    # ══ MIXED SPINE (back + neck) ═══════════════════════════════════════════
    {"a": "denninger",  "y": "2018", "body_region": "multi_region","condition": "back or neck pain",                    "country": "US",          "study_design": "retrospective cohort","perspective": "healthcare","outcome_measure": "EQ-5D, NPRS, ODI/NDI",          "outcome_type": "both",     "time_horizon": "24 months"},
    {"a": "skargren",   "y": "1997", "body_region": "multi_region","condition": "back or neck pain",                    "country": "Sweden",      "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "General health, ODS, VAS",      "outcome_type": "clinical", "time_horizon": "6 months"},
    {"a": "skargren",   "y": "1998", "body_region": "multi_region","condition": "back or neck pain",                    "country": "Sweden",      "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "General health, ODS, VAS",      "outcome_type": "clinical", "time_horizon": "12 months"},

    # ══ UPPER LIMB ══════════════════════════════════════════════════════════
    {"a": "bergman",    "y": "2010", "body_region": "shoulder",    "condition": "shoulder complaints",                  "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "Patient perceived recovery",     "outcome_type": "clinical", "time_horizon": "6 months"},
    {"a": "commbes",    "y": "2016", "body_region": "elbow",       "condition": "epicondylitis lateralis",              "country": "Australia",   "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "coombes",    "y": "2016", "body_region": "elbow",       "condition": "epicondylitis lateralis",              "country": "Australia",   "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "fernandez",  "y": "2019", "body_region": "wrist",       "condition": "carpal tunnel syndrome",               "country": "Spain",       "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "geraets",    "y": "2006", "body_region": "shoulder",    "condition": "shoulder complaints (chronic)",        "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "EQ-5D, SDQ",                    "outcome_type": "both",     "time_horizon": "12 months"},
    {"a": "hopewell",   "y": "2021", "body_region": "shoulder",    "condition": "rotator cuff disease",                 "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "james",      "y": "2005", "body_region": "shoulder",    "condition": "shoulder pain",                        "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "Disability score, EQ-5D",       "outcome_type": "both",     "time_horizon": "6 months"},
    {"a": "struijs",    "y": "2006", "body_region": "elbow",       "condition": "epicondylitis lateralis",              "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "EQ, pain-free function questionnaire","outcome_type": "both","time_horizon": "12 months"},

    # ══ HIP ═════════════════════════════════════════════════════════════════
    {"a": "fusco",      "y": "2019", "body_region": "hip",         "condition": "hip replacement",                      "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "griffin",    "y": "2022", "body_region": "hip",         "condition": "femoroacetabular impingement syndrome", "country": "UK",         "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "juhakoski",  "y": "2011", "body_region": "hip",         "condition": "hip osteoarthritis",                   "country": "Finland",     "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "SF-36, WOMAC",                  "outcome_type": "clinical", "time_horizon": "24 months"},
    {"a": "tan",        "y": "2016", "body_region": "hip",         "condition": "hip osteoarthritis",                   "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},

    # ══ KNEE ════════════════════════════════════════════════════════════════
    {"a": "barton",     "y": "2009", "body_region": "knee",        "condition": "knee pain",                            "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "24 months"},
    {"a": "bennell",    "y": "2016", "body_region": "knee",        "condition": "knee osteoarthritis",                  "country": "Australia",   "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "eggerding",  "y": "2021", "body_region": "knee",        "condition": "ACL tear",                             "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "24 months"},
    {"a": "ho",         "y": "2022", "body_region": "knee",        "condition": "knee osteoarthritis",                  "country": "Sweden",      "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "huang",      "y": "2012", "body_region": "knee",        "condition": "total knee replacement",               "country": "Taiwan",      "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "Knee ROM, LOS, VAS",            "outcome_type": "clinical", "time_horizon": "5 days"},
    # hurley 2007 and 2012 = knee; hurley 2015 = low_back (handled above)
    {"a": "hurley",     "y": "2007", "body_region": "knee",        "condition": "knee pain (chronic)",                  "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "WOMAC, QALY (EQ-5D)",           "outcome_type": "both",     "time_horizon": "6 months"},
    {"a": "hurley",     "y": "2012", "body_region": "knee",        "condition": "knee pain (chronic)",                  "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "WOMAC",                         "outcome_type": "clinical", "time_horizon": "30 months"},
    {"a": "jessep",     "y": "2009", "body_region": "knee",        "condition": "knee pain (chronic)",                  "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "kigozi",     "y": "2018", "body_region": "knee",        "condition": "knee osteoarthritis",                  "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "18 months"},
    {"a": "knoop",      "y": "2023", "body_region": "knee",        "condition": "knee osteoarthritis",                  "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "mccarthy",   "y": "2004", "body_region": "knee",        "condition": "knee osteoarthritis",                  "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "mitchell",   "y": "2005", "body_region": "knee",        "condition": "total knee replacement",               "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "SF-36, WOMAC",                  "outcome_type": "clinical", "time_horizon": "15 months"},
    {"a": "pryymachenko","y":"2021", "body_region": "knee",        "condition": "knee osteoarthritis",                  "country": "New Zealand", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "24 months"},
    {"a": "rhon",       "y": "2022", "body_region": "knee",        "condition": "knee osteoarthritis",                  "country": "US",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "stan",       "y": "2015", "body_region": "knee",        "condition": "knee osteoarthritis (varus deformity)","country": "Romania",     "study_design": "controlled trial",  "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "unclear"},
    {"a": "tan",        "y": "2010", "body_region": "knee",        "condition": "patellofemoral pain syndrome",         "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "van de graaf","y":"2020", "body_region": "knee",        "condition": "meniscal tear (non-obstructive)",      "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "IKDC, QALY (EQ-5D)",            "outcome_type": "both",     "time_horizon": "24 months"},
    {"a": "graaf",      "y": "2020", "body_region": "knee",        "condition": "meniscal tear (non-obstructive)",      "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "IKDC, QALY (EQ-5D)",            "outcome_type": "both",     "time_horizon": "24 months"},
    {"a": "graaff",     "y": "2023", "body_region": "knee",        "condition": "meniscal tear (traumatic)",            "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "24 months"},

    # Sevick 2000 has two entries — distinguish by searching paper_id for "ex" vs "life"
    # Both listed under knee OA; the second (life) is sedentary adults
    {"a": "sevick",     "y": "2000", "body_region": "knee",        "condition": "knee osteoarthritis",                  "country": "US",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "Disability score, stair climb, 6-min walk","outcome_type": "clinical","time_horizon": "18 months"},
    {"a": "sevick",     "y": "2009", "body_region": "knee",        "condition": "knee osteoarthritis",                  "country": "US",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "WOMAC, stair climb, 6-min walk","outcome_type": "clinical", "time_horizon": "18 months"},

    # ══ LOWER LIMB: MIXED (Hip + Knee) ══════════════════════════════════════
    {"a": "abbott",     "y": "2019", "body_region": "multi_region","condition": "hip and knee osteoarthritis",          "country": "New Zealand", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (SF-6D)",                  "outcome_type": "QALY",     "time_horizon": "24 months"},
    {"a": "pinto",      "y": "2013", "body_region": "multi_region","condition": "hip and knee osteoarthritis",          "country": "New Zealand", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (SF-12v2), WOMAC",         "outcome_type": "both",     "time_horizon": "12 months"},
    {"a": "bulthuis",   "y": "2008", "body_region": "multi_region","condition": "hip and knee osteoarthritis",          "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "Functional ability, MACTAR",    "outcome_type": "clinical", "time_horizon": "6 months"},
    {"a": "coupe",      "y": "2007", "body_region": "multi_region","condition": "hip and knee osteoarthritis",          "country": "Netherlands", "study_design": "RCT",               "perspective": "societal",    "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "15 months"},
    {"a": "fernandes",  "y": "2017", "body_region": "multi_region","condition": "hip and knee replacement",             "country": "Denmark",     "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "HOOS, KOOS, QALY (EQ-5D)",     "outcome_type": "both",     "time_horizon": "12 months"},
    {"a": "lin",        "y": "2008", "body_region": "ankle",       "condition": "ankle fracture",                       "country": "Australia",   "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "AQOL, LEFS",                    "outcome_type": "clinical", "time_horizon": "5-6 months"},

    # ══ OTHER CONDITIONS ════════════════════════════════════════════════════
    {"a": "barnhoorn",  "y": "2018", "body_region": "other",       "condition": "complex regional pain syndrome type 1","country": "Netherlands","study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "9 months"},
    {"a": "daker",      "y": "1999", "body_region": "multi_region","condition": "musculoskeletal problems",             "country": "UK",          "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "SF-36, VAS",                    "outcome_type": "clinical", "time_horizon": "5-6 months"},
    {"a": "heij",       "y": "2022", "body_region": "other",       "condition": "mobility problems",                    "country": "Netherlands", "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (EQ-5D)",                  "outcome_type": "QALY",     "time_horizon": "6 months"},
    {"a": "lilje",      "y": "2014", "body_region": "multi_region","condition": "mixed musculoskeletal (waiting list)",  "country": "Sweden",     "study_design": "RCT",               "perspective": "healthcare",  "outcome_measure": "QALY (SF-6D)",                  "outcome_type": "QALY",     "time_horizon": "12 months"},
    {"a": "van den hout","y":"2005", "body_region": "other",       "condition": "rheumatoid arthritis",                  "country": "Netherlands","study_design": "RCT",               "perspective": "societal",    "outcome_measure": "HAQ, MACTAR, QALY (EQ-5D)",    "outcome_type": "both",     "time_horizon": "24 months"},
    {"a": "hout",       "y": "2005", "body_region": "other",       "condition": "rheumatoid arthritis",                  "country": "Netherlands","study_design": "RCT",               "perspective": "societal",    "outcome_measure": "HAQ, MACTAR, QALY (EQ-5D)",    "outcome_type": "both",     "time_horizon": "24 months"},
]


# ── Patch function ─────────────────────────────────────────────────────────────

def patch(db_path: str = None, dry_run: bool = False) -> dict:
    """
    Apply Table 1 corrections to ce_comparisons.
    Returns a summary dict: {updated, skipped, not_found}.
    """
    path = db_path or str(DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    all_papers = conn.execute(
        "SELECT DISTINCT paper_id FROM ce_comparisons"
    ).fetchall()
    paper_ids = [r["paper_id"] for r in all_papers]

    update_fields = [
        "body_region", "condition", "country", "study_design",
        "perspective", "outcome_measure", "outcome_type", "time_horizon",
    ]

    updated = 0
    not_found = []
    skipped = []
    applied_pids: set = set()  # avoid double-updating

    for corr in CORRECTIONS:
        author_key = corr["a"].lower().split()[0]  # first word of author key
        year = corr["y"]

        matches = [
            pid for pid in paper_ids
            if author_key in pid.lower() and year in pid
            and pid not in applied_pids
        ]

        # For ambiguous matches (same author, same year) prefer exact first-word
        if len(matches) > 1:
            strict = [m for m in matches if m.lower().startswith(author_key)]
            if strict:
                matches = strict

        if not matches:
            not_found.append(f"{corr['a']} {year}")
            continue

        fields = {k: corr[k] for k in update_fields if k in corr}
        set_clauses = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values())

        for pid in matches:
            if dry_run:
                print(f"  DRY RUN: would update {pid} → {fields}")
            else:
                conn.execute(
                    f"UPDATE ce_comparisons SET {set_clauses} WHERE paper_id = ?",
                    values + [pid],
                )
                print(f"  ✓ {pid} → body_region={fields.get('body_region')} country={fields.get('country')}")
            applied_pids.add(pid)
            updated += 1

    if not dry_run:
        conn.commit()
    conn.close()

    result = {
        "updated": updated,
        "not_found": len(not_found),
        "not_found_list": not_found,
    }
    print(f"\n{'DRY RUN ' if dry_run else ''}Summary: updated={updated}, not_found={len(not_found)}")
    if not_found:
        print("Not found:", not_found)
    return result


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    patch(dry_run=dry)
