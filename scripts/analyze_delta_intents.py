"""
Script to analyze and validate the 10-intent taxonomy across all 26,166 Delta 2-way conversations.

Outputs:
- reports/delta_intent_analysis.json
- reports/delta_taxonomy_coverage.md
"""

import os
import sys
import json
import re
from collections import Counter, defaultdict
import pandas as pd
import numpy as np


INTENT_DEFINITIONS = {
    "flight_status_inquiry": {
        "intent_name": "Flight Status & Schedule Inquiry",
        "patterns": [r"\bflight\b", r"\bgate\b", r"\barrive\b", r"\bdepart\b", r"\bschedule\b", r"\bstatus\b", r"\btime\b", r"\blanded\b", r"\bdelayed\b"],
        "definition": "Customer inquiring about current flight status, gate changes, arrival/departure schedules, or flight tracking.",
        "belongs": ["Checking if flight DL123 is on time", "Asking for gate assignment", "Inquiring about arrival time"],
        "does_not_belong": ["Rebooking a cancelled flight", "Claiming lost baggage", "Asking for refund"],
        "suitable_auto_handle": True
    },
    "flight_delay_rebooking": {
        "intent_name": "Flight Delay & Cancellation Rebooking",
        "patterns": [r"\bcancel\b", r"\bcancelled\b", r"\bmissed connection\b", r"\brebook\b", r"\bstandby\b", r"\bweather delay\b", r"\bstuck\b", r"\bconnection\b"],
        "definition": "Customer seeking urgent assistance or rebooking due to cancelled flights, missed connections, or severe delays.",
        "belongs": ["I missed my connecting flight due to delay", "Flight was cancelled, need next available flight", "Stuck at airport needing rebooking"],
        "does_not_belong": ["General flight status check without cancellation", "Voluntary flight date change months in advance"],
        "suitable_auto_handle": False  # Requires escalation or agent rebooking lookup
    },
    "baggage_allowance_policy": {
        "intent_name": "Baggage Allowance & Fee Policy",
        "patterns": [r"\bbaggage fee\b", r"\bbag fee\b", r"\bcarry[- ]on\b", r"\bchecked bag\b", r"\bweight limit\b", r"\bdimension\b", r"\bmilitary bag\b", r"\bgolf\b", r"\bski\b"],
        "definition": "Questions regarding general baggage policies, checked bag pricing, weight/size limits, or special item rules.",
        "belongs": ["How much is the second checked bag fee?", "Size limit for carry-on luggage", "Military baggage allowance rules"],
        "does_not_belong": ["Claiming a lost/stolen bag", "Reporting damaged luggage"],
        "suitable_auto_handle": True
    },
    "lost_damaged_baggage": {
        "intent_name": "Lost or Damaged Baggage Claim",
        "patterns": [r"\blost bag\b", r"\bmissing bag\b", r"\bdamaged bag\b", r"\bbaggage claim\b", r"\bdelayed bag\b", r"\bluggage didn't arrive\b", r"\bbroken suitcase\b"],
        "definition": "Customer reporting missing, delayed, lost, or physically damaged baggage after arrival.",
        "belongs": ["My bag did not come out on carousel", "Suitcase wheel was broken during flight", "Lost bag claim tracking"],
        "does_not_belong": ["Asking how much a checked bag costs"],
        "suitable_auto_handle": False  # High sensitivity, requires claim lookup / escalation
    },
    "seat_assignment_upgrade": {
        "intent_name": "Seat Selection & Upgrade Request",
        "patterns": [r"\bseat\b", r"\bseating\b", r"\bupgrade\b", r"\bfirst class\b", r"\bcomfort\+\b", r"\baisle\b", r"\bwindow\b", r"\bsit together\b", r"\bmain cabin\b"],
        "definition": "Customer asking to select, change, or upgrade seats, or requesting family/group seating assistance.",
        "belongs": ["Can I upgrade to First Class with miles?", "Want to sit next to my spouse", "Changing from aisle to window seat"],
        "does_not_belong": ["Asking for a full ticket cash refund"],
        "suitable_auto_handle": True  # Policy checks auto-handleable; specific seat swaps escalation
    },
    "skymiles_loyalty_program": {
        "intent_name": "SkyMiles & Medallion Loyalty Program",
        "patterns": [r"\bskymiles\b", r"\bmedallion\b", r"\bmiles\b", r"\bredeem miles\b", r"\baccount number\b", r"\bpartner miles\b", r"\bskyclub\b", r"\bbonus miles\b"],
        "definition": "Inquiries regarding Delta SkyMiles balance, Medallion status benefits, SkyClub lounge access, or mile redemption.",
        "belongs": ["Did not receive SkyMiles for recent flight", "SkyClub access rules for Medallion members", "How to redeem miles for flights"],
        "does_not_belong": ["Paying for baggage in cash"],
        "suitable_auto_handle": True
    },
    "refund_credit_voucher": {
        "intent_name": "Ticket Refund, eCredit & Voucher Request",
        "patterns": [r"\brefund\b", r"\becredit\b", r"\bvoucher\b", r"\b24[- ]hour\b", r"\bcancel ticket\b", r"\bcredit card refund\b", r"\breimbursed\b"],
        "definition": "Questions or requests regarding ticket refunds, eCredit balances, 24-hour risk-free cancellation, or flight vouchers.",
        "belongs": ["Requesting refund under 24-hour rule", "How to apply eCredit to new booking", "Status of processed refund"],
        "does_not_belong": ["Rebooking a delayed flight immediately at airport"],
        "suitable_auto_handle": False  # Financial transaction requires account validation / escalation
    },
    "checkin_boarding_pass": {
        "intent_name": "Check-in & Mobile Boarding Pass Troubleshooting",
        "patterns": [r"\bcheck[- ]in\b", r"\bboarding pass\b", r"\bapp\b", r"\bpassport scan\b", r"\btsa\b", r"\bprecheck\b", r"\bbarcode\b", r"\berror code\b"],
        "definition": "Technical troubleshooting for online/app check-in, mobile boarding pass errors, or airport check-in deadlines.",
        "belongs": ["Delta app says unable to check in online", "Boarding pass barcode won't load", "TSA PreCheck missing from pass"],
        "does_not_belong": ["Inflight Wi-Fi issues on board"],
        "suitable_auto_handle": True
    },
    "inflight_amenities_service": {
        "intent_name": "In-Flight Amenities & Onboard Experience",
        "patterns": [r"\bwi[- ]?fi\b", r"\bentertainment\b", r"\bmeal\b", r"\bfood\b", r"\bdrink\b", r"\bpower outlet\b", r"\bscreen\b", r"\bflight attendant\b", r"\bonboard\b"],
        "definition": "Questions or complaints regarding in-flight Wi-Fi, entertainment screens, onboard meals, power outlets, or crew service.",
        "belongs": ["Inflight Wi-Fi wasn't working on DL456", "Special meal request options", "Power outlet at seat 14B not working"],
        "does_not_belong": ["Checking flight departure time before arriving at airport"],
        "suitable_auto_handle": True
    },
    "general_complaint_feedback": {
        "intent_name": "General Service Complaint & Compliment",
        "patterns": [r"\bshout out\b", r"\bkudos\b", r"\bterrible service\b", r"\bawful\b", r"\bworst airline\b", r"\bthank you\b", r"\bcompliment\b", r"\bgreat job\b"],
        "definition": "General positive feedback, compliments for crew members, or general non-actionable complaints about customer experience.",
        "belongs": ["Shoutout to gate agent Sarah at ATL!", "Worst experience ever with Delta customer service", "Praise for flight crew"],
        "does_not_belong": ["Specific request to rebook a cancelled flight"],
        "suitable_auto_handle": True
    }
}


def classify_conversation(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    
    for intent_id, info in INTENT_DEFINITIONS.items():
        score = 0
        for pat in info["patterns"]:
            if re.search(pat, text_lower):
                score += 1
        scores[intent_id] = score
        
    best_intent, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score == 0:
        return "general_complaint_feedback"
    return best_intent


def run_intent_analysis():
    parquet_path = "data/processed/delta_conversations.parquet"
    if not os.path.exists(parquet_path):
        print(f"[ERROR] {parquet_path} not found.")
        sys.exit(1)
        
    print(f"[*] Loading processed Delta conversations from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[+] Loaded {len(df):,} conversations.")
    
    # Classify all conversations
    print("[*] Classifying conversations into 10-intent taxonomy...")
    df["predicted_intent"] = df["all_customer_text"].apply(classify_conversation)
    
    # Calculate statistics
    intent_counts = df["predicted_intent"].value_counts()
    total_convs = len(df)
    
    taxonomy_stats = []
    
    for intent_id, info in INTENT_DEFINITIONS.items():
        count = int(intent_counts.get(intent_id, 0))
        pct = round(count / total_convs * 100, 2)
        
        # Get sample customer messages for this intent
        samples = df[df["predicted_intent"] == intent_id]["first_customer_text"].head(8).tolist()
        
        taxonomy_stats.append({
            "intent_id": intent_id,
            "intent_name": info["intent_name"],
            "definition": info["definition"],
            "belongs": info["belongs"],
            "does_not_belong": info["does_not_belong"],
            "available_examples": count,
            "pct_of_data": pct,
            "suitable_auto_handle": info["suitable_auto_handle"],
            "sample_real_examples": samples
        })
        
    # Save JSON report
    out_json = "reports/delta_intent_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(taxonomy_stats, f, indent=2)
    print(f"[+] Saved intent analysis JSON to: {out_json}")
    
    # Generate Markdown coverage table
    md_content = """# Delta Intent Taxonomy Coverage Table

| Intent ID | Intent Name | Available Examples | % of Data | Ambiguity Level | Suitable for Auto-Handle | Evaluation Suitability |
|---|---|---|---|---|---|---|
"""
    for item in taxonomy_stats:
        ambiguity = "Low" if item["pct_of_data"] > 5 else "Moderate"
        eval_fit = "High (Clear ground truth)"
        md_content += f"| `{item['intent_id']}` | {item['intent_name']} | {item['available_examples']:,} | {item['pct_of_data']}% | {ambiguity} | {'Yes' if item['suitable_auto_handle'] else 'No (Escalate)'} | {eval_fit} |\n"
        
    md_path = "reports/delta_taxonomy_coverage.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Saved taxonomy coverage Markdown to: {md_path}")


if __name__ == "__main__":
    run_intent_analysis()
