#!/usr/bin/env python3
"""Phase 4: Generate construction-verified synthetic training data for weak types.

Reads the error analysis report and generates targeted training records with
ground-truth spans computed by construction (same approach as build_gold.py).
This is even stronger than teacher verification — offsets are exact because
we control PII insertion.

Uses a DIFFERENT Faker seed than the gold set (42) to ensure zero overlap.

Usage:
    python scripts/generate_targeted_seeds.py \
        --output data/targeted_train.jsonl \
        --seed 1337

    # With error analysis report for auto-weighting:
    python scripts/generate_targeted_seeds.py \
        --output data/targeted_train.jsonl \
        --error-report data/error_analysis_run001.json
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from forge.schema import HIGH_SEVERITY, PIIRecord, PIIType

# Reuse the gold set builder's infrastructure
from scripts.build_gold import PIIValueGenerator, build_record, expand_template, print_stats

P = PIIType

# fmt: off
# =====================================================================
# Targeted templates — weighted heavily toward weak types
# Each entry: (template, weight)
# =====================================================================

TARGETED_TEMPLATES: list[tuple[list[str | PIIType], int]] = [
    # =================================================================
    # A. MULTI-PERSON (biggest gap: 267 PERSON FN, model only finds ~1 per record)
    # =================================================================
    (["Team members: ", P.PERSON, ", ", P.PERSON, ", and ", P.PERSON, "."], 12),
    (["Meeting between ", P.PERSON, " and ", P.PERSON, " at 3pm."], 10),
    ([P.PERSON, " introduced ", P.PERSON, " to ", P.PERSON, " at the conference."], 8),
    (["CC: ", P.PERSON, ", ", P.PERSON, ". Subject: project update."], 8),
    ([P.PERSON, " and ", P.PERSON, " co-authored the report."], 8),
    (["Attendees: ", P.PERSON, ", ", P.PERSON, ", ", P.PERSON, ", ", P.PERSON, "."], 6),
    (["Interview panel: ", P.PERSON, " (lead), ", P.PERSON, ", ", P.PERSON, "."], 6),
    (["Dear ", P.PERSON, ", please coordinate with ", P.PERSON, " on this matter."], 6),
    ([P.PERSON, " emailed ", P.PERSON, " about the deadline."], 6),
    (["Witnesses: ", P.PERSON, " and ", P.PERSON, ". Filed by ", P.PERSON, "."], 6),
    ([P.PERSON, " approved the request submitted by ", P.PERSON, "."], 5),
    (["From ", P.PERSON, " to ", P.PERSON, ": please review before Friday."], 5),
    ([P.PERSON, ", ", P.PERSON, ", and ", P.PERSON, " joined the call."], 5),
    (["Mentored by ", P.PERSON, ". Mentee: ", P.PERSON, "."], 5),

    # PERSON + other types (contextual multi-entity)
    ([P.PERSON, " (", P.EMAIL, ") referred ", P.PERSON, " for the role."], 5),
    (["Contact ", P.PERSON, " at ", P.PHONE, " or ", P.PERSON, " at ", P.EMAIL, "."], 5),
    ([P.PERSON, " lives at ", P.STREET_ADDRESS, " with ", P.PERSON, "."], 4),
    ([P.PERSON, ", born ", P.DATE_OF_BIRTH, ", married to ", P.PERSON, "."], 4),

    # =================================================================
    # B. AADHAAR (need ~53 more, 6.9% recall, critical)
    # =================================================================
    (["Aadhaar number: ", P.AADHAAR, ". Holder: ", P.PERSON, "."], 8),
    (["KYC: Aadhaar ", P.AADHAAR, " linked to PAN ", P.PAN, "."], 7),
    (["Verification status: Aadhaar ", P.AADHAAR, " confirmed."], 7),
    ([P.PERSON, "'s Aadhaar is ", P.AADHAAR, "."], 7),
    (["Aadhaar card: ", P.AADHAAR, ". Address: ", P.STREET_ADDRESS, "."], 6),
    (["Your Aadhaar ", P.AADHAAR, " has been seeded with your bank account."], 6),
    (["eKYC via Aadhaar: ", P.AADHAAR, ". Mobile: ", P.PHONE, "."], 5),
    (["Aadhaar enrollment: ", P.AADHAAR, " for ", P.PERSON, " of ", P.LOCATION, "."], 5),
    (["Update Aadhaar ", P.AADHAAR, " details at the nearest centre."], 5),
    (["Aadhaar-PAN linking: ", P.AADHAAR, " <-> ", P.PAN, ". Status: linked."], 4),

    # =================================================================
    # C. PAN (need ~51 more, 10.3% recall, critical)
    # =================================================================
    (["PAN: ", P.PAN, ". Name: ", P.PERSON, "."], 8),
    (["Tax return filed with PAN ", P.PAN, " for AY 2025-26."], 7),
    (["PAN ", P.PAN, " linked to bank account ", P.BANK_ACCOUNT, "."], 6),
    ([P.PERSON, " has PAN number ", P.PAN, "."], 7),
    (["Form 26AS for PAN: ", P.PAN, ". TDS: Rs 45,000."], 6),
    (["ITR verification: PAN ", P.PAN, ", Aadhaar ", P.AADHAAR, "."], 5),
    (["PAN card details: ", P.PAN, ", DOB: ", P.DATE_OF_BIRTH, "."], 5),
    (["Investor PAN: ", P.PAN, ". Folio: 1234567890."], 5),
    (["GST registration under PAN ", P.PAN, " approved."], 5),
    (["PAN ", P.PAN, " is required for transactions above Rs 50,000."], 5),

    # =================================================================
    # D. DRIVER_LICENSE (need ~29 more, 0% recall, critical)
    # =================================================================
    (["Driver license: ", P.DRIVER_LICENSE, ". Holder: ", P.PERSON, "."], 8),
    (["DL number ", P.DRIVER_LICENSE, " valid until 2029."], 7),
    (["Identity proof: driver license ", P.DRIVER_LICENSE, "."], 7),
    ([P.PERSON, " holds DL ", P.DRIVER_LICENSE, "."], 7),
    (["Traffic challan for DL ", P.DRIVER_LICENSE, ". Fine: Rs 500."], 6),
    (["Rental agreement: DL ", P.DRIVER_LICENSE, ", tenant ", P.PERSON, "."], 5),
    (["DL ", P.DRIVER_LICENSE, " issued in ", P.LOCATION, "."], 5),
    (["License ", P.DRIVER_LICENSE, " suspended for 6 months."], 5),
    (["Vehicle registration: owner DL ", P.DRIVER_LICENSE, ", address ", P.STREET_ADDRESS, "."], 4),

    # =================================================================
    # E. CREDIT_CARD (need ~55 more, 31.7% recall, critical)
    # =================================================================
    (["Card number: ", P.CREDIT_CARD, ". Holder: ", P.PERSON, "."], 7),
    (["Charge of Rs 5,999 on card ", P.CREDIT_CARD, "."], 7),
    (["Refund to card ending in ", P.CREDIT_CARD, " processed."], 6),
    ([P.PERSON, " paid with card ", P.CREDIT_CARD, "."], 6),
    (["Auto-debit on card ", P.CREDIT_CARD, " for subscription renewal."], 6),
    (["Card ", P.CREDIT_CARD, " blocked. Contact ", P.PHONE, "."], 5),
    (["New card issued: ", P.CREDIT_CARD, ". Replace old card immediately."], 5),
    (["Payment failed on card ", P.CREDIT_CARD, ". Try ", P.CREDIT_CARD, "."], 4),
    (["Card ", P.CREDIT_CARD, " linked to account ", P.BANK_ACCOUNT, "."], 5),

    # =================================================================
    # F. BANK_ACCOUNT (need ~43 more, 24.1% recall, critical)
    # =================================================================
    (["Bank account: ", P.BANK_ACCOUNT, ". IFSC: SBIN0001234."], 7),
    (["Transfer to account ", P.BANK_ACCOUNT, " completed."], 7),
    (["Salary credited to ", P.BANK_ACCOUNT, " for ", P.PERSON, "."], 6),
    ([P.PERSON, "'s account number is ", P.BANK_ACCOUNT, "."], 6),
    (["Debit of Rs 10,000 from account ", P.BANK_ACCOUNT, "."], 6),
    (["Account ", P.BANK_ACCOUNT, " linked to Aadhaar ", P.AADHAAR, "."], 5),
    (["Beneficiary account: ", P.BANK_ACCOUNT, ". Name: ", P.PERSON, "."], 5),
    (["Close account ", P.BANK_ACCOUNT, "? Remaining balance: Rs 1,234."], 5),
    (["EMI auto-debit from account ", P.BANK_ACCOUNT, " on 5th of each month."], 5),

    # =================================================================
    # G. PASSPORT (need ~14 more, 78.3% recall, medium)
    # =================================================================
    (["Passport: ", P.PASSPORT, ". Nationality: Indian."], 6),
    ([P.PERSON, " holds passport ", P.PASSPORT, ". Expiry: 2030-06-15."], 5),
    (["Visa application: passport ", P.PASSPORT, ", DOB ", P.DATE_OF_BIRTH, "."], 5),
    (["Immigration: passport ", P.PASSPORT, " scanned at checkpoint."], 5),
    (["Passport ", P.PASSPORT, " and DL ", P.DRIVER_LICENSE, " submitted for verification."], 4),

    # =================================================================
    # H. PASSWORD (need ~23 more, 36.8% recall, critical)
    # =================================================================
    (["Your password is: ", P.PASSWORD, ". Change it now."], 7),
    (["Temporary password: ", P.PASSWORD, ". User: ", P.USERNAME, "."], 6),
    (["Reset password from ", P.PASSWORD, " to a stronger one."], 6),
    ([P.PERSON, " set their password to ", P.PASSWORD, "."], 6),
    (["Default password: ", P.PASSWORD, ". Must change on first login."], 5),
    (["Password for WiFi: ", P.PASSWORD, "."], 5),
    (["Admin password: ", P.PASSWORD, ". API key: ", P.API_KEY, "."], 4),

    # =================================================================
    # I. API_KEY (need ~33 more, 41.4% recall, critical)
    # =================================================================
    (["API key: ", P.API_KEY, ". Keep it secret."], 7),
    (["Use token ", P.API_KEY, " for authentication."], 6),
    (["Revoked API key: ", P.API_KEY, ". Generate a new one."], 6),
    (["Config: api_key=", P.API_KEY, "."], 6),
    ([P.USERNAME, " created API key ", P.API_KEY, "."], 5),
    (["Service token: ", P.API_KEY, ". Endpoint: ", P.URL, "."], 5),
    (["Production key: ", P.API_KEY, ". Staging key: ", P.API_KEY, "."], 4),

    # =================================================================
    # J. STREET_ADDRESS (need ~49, 12.5% recall, high)
    # =================================================================
    (["Address: ", P.STREET_ADDRESS, ". Recipient: ", P.PERSON, "."], 7),
    (["Deliver to ", P.STREET_ADDRESS, "."], 6),
    (["Office location: ", P.STREET_ADDRESS, "."], 6),
    ([P.PERSON, " lives at ", P.STREET_ADDRESS, "."], 6),
    (["Billing address: ", P.STREET_ADDRESS, ". Card: ", P.CREDIT_CARD, "."], 5),
    (["Registered address: ", P.STREET_ADDRESS, ". PAN: ", P.PAN, "."], 4),
    (["Warehouse at ", P.STREET_ADDRESS, ". Contact: ", P.PHONE, "."], 4),
    (["Return to: ", P.STREET_ADDRESS, ". Attention: ", P.PERSON, "."], 5),

    # =================================================================
    # K. LOCATION (need ~23, 36.4% recall, high)
    # =================================================================
    ([P.PERSON, " relocated to ", P.LOCATION, "."], 6),
    (["Branch office in ", P.LOCATION, ". Manager: ", P.PERSON, "."], 5),
    (["Event in ", P.LOCATION, ". Register at ", P.URL, "."], 5),
    ([P.PERSON, " is based in ", P.LOCATION, " and works remotely."], 5),
    (["Transferred from ", P.LOCATION, " to ", P.LOCATION, "."], 4),

    # =================================================================
    # L. USERNAME (need ~49, 31.0% recall, high)
    # =================================================================
    (["User ", P.USERNAME, " logged in successfully."], 6),
    (["Account locked: ", P.USERNAME, ". Contact admin."], 6),
    (["Profile: ", P.USERNAME, " (", P.EMAIL, ")."], 5),
    (["Assigned to ", P.USERNAME, " by ", P.PERSON, "."], 5),
    ([P.USERNAME, " posted a comment at ", P.URL, "."], 5),
    (["Permissions updated for ", P.USERNAME, " on server ", P.IP_ADDRESS, "."], 4),
    (["Deactivating user ", P.USERNAME, ". Reason: inactivity."], 4),

    # =================================================================
    # M. AGE (need ~12, 55.6% recall, high)
    # =================================================================
    ([P.PERSON, ", ", P.AGE, " years old, applied for the position."], 5),
    (["Applicant age: ", P.AGE, ". Name: ", P.PERSON, "."], 5),
    ([P.PERSON, " is ", P.AGE, " and lives in ", P.LOCATION, "."], 4),

    # =================================================================
    # N. COMPLEX MULTI-TYPE (teaches multi-span extraction)
    # =================================================================
    (["KYC: ", P.PERSON, ", Aadhaar ", P.AADHAAR, ", PAN ", P.PAN, ", phone ", P.PHONE, "."], 6),
    ([P.PERSON, " (DL ", P.DRIVER_LICENSE, ") at ", P.STREET_ADDRESS, ", card ", P.CREDIT_CARD, "."], 5),
    (["Onboarding: ", P.PERSON, ", passport ", P.PASSPORT, ", email ", P.EMAIL, ", DOB ", P.DATE_OF_BIRTH, "."], 5),
    (["Support: ", P.USERNAME, " (", P.EMAIL, ") from ", P.IP_ADDRESS, ", password '", P.PASSWORD, "'."], 5),
    (["Banking: ", P.PERSON, ", account ", P.BANK_ACCOUNT, ", PAN ", P.PAN, ", Aadhaar ", P.AADHAAR, "."], 5),
    (["Payroll: ", P.PERSON, ", age ", P.AGE, ", card ", P.CREDIT_CARD, ", account ", P.BANK_ACCOUNT, "."], 4),
    ([P.PERSON, ", passport ", P.PASSPORT, ", DL ", P.DRIVER_LICENSE, ", address ", P.STREET_ADDRESS, "."], 4),
    (["IT alert: ", P.USERNAME, " at ", P.IP_ADDRESS, " used key ", P.API_KEY, " and password ", P.PASSWORD, "."], 4),
    (["Applicant: ", P.PERSON, " (", P.AGE, "), PAN ", P.PAN, ", email ", P.EMAIL, ", in ", P.LOCATION, "."], 4),

    # =================================================================
    # O. HARD NEGATIVES (reduce FP for SSN, STREET_ADDRESS, USERNAME)
    # =================================================================
    (["Order #123-456-7890 has been shipped to ", P.PERSON, "."], 5),
    (["Reference number: 987-65-4321. Please quote this in correspondence."], 4),
    (["PIN code: 400001. Area: South Mumbai."], 4),
    (["The meeting room capacity is 25. Book via the intranet."], 4),
    (["Version 3.2.1 released on 2025-06-15. Update now."], 3),
    (["Invoice #INV-2025-0042 for Rs 12,500 is due by month end."], 3),
    (["Flat 402, Tower B is available for rent. Contact property manager."], 3),
    (["The report covers data from 2020 to 2024 across all regions."], 3),
    (["Sprint velocity: 42 points. Next standup at 9:30am."], 3),
    (["Port 8080 is open on the staging server. Close after testing."], 3),
]
# fmt: on


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate targeted synthetic training data.")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL")
    ap.add_argument("--seed", type=int, default=1337, help="Faker seed (must differ from gold set's 42)")
    ap.add_argument("--error-report", type=Path, default=None, help="Error analysis JSON (unused — weights are baked in)")
    ap.add_argument("--dedup-against", type=Path, nargs="*", default=[], help="JSONL files to check for text overlap")
    args = ap.parse_args()

    if args.seed == 42:
        print("ERROR: seed 42 is reserved for the gold set. Use a different seed.", file=sys.stderr)
        return 1

    gen = PIIValueGenerator(args.seed)
    rng = random.Random(args.seed + 1)

    existing_texts: set[str] = set()
    for dedup_path in args.dedup_against:
        p = Path(dedup_path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = PIIRecord.model_validate_json(line)
                    existing_texts.add(r.text)

    records: list[PIIRecord] = []
    seq = 0
    for tpl, weight in TARGETED_TEMPLATES:
        for _ in range(weight):
            seq += 1
            segments = expand_template(tpl, gen)
            record = build_record(f"aug-{seq:04d}", segments, split="train")
            if record.text not in existing_texts:
                records.append(record)

    rng.shuffle(records)

    for i, r in enumerate(records, 1):
        records[i - 1] = r.model_copy(update={"id": f"aug-{i:04d}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    print(f"generated {len(records)} targeted training records -> {args.output}")
    print_stats(records, "TARGETED AUGMENTATION")

    type_counts: Counter = Counter()
    for r in records:
        for s in r.spans:
            type_counts[s.label.value] += 1

    print("\nSpan distribution:")
    for t in PIIType:
        c = type_counts.get(t.value, 0)
        if c > 0:
            hs = " [HIGH-SEV]" if t in HIGH_SEVERITY else ""
            print(f"  {t.value:<20} {c:>5}{hs}")
    print(f"  {'TOTAL':<20} {sum(type_counts.values()):>5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
