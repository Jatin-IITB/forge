#!/usr/bin/env python3
"""Build the frozen gold evaluation set (dev.jsonl + test.jsonl).

Uses Faker (MIT) with a fixed seed to inject synthetic PII values into diverse
carrier text templates. Ground-truth spans are exact by construction — offsets
are computed from the segment list, never hand-counted.

See data/gold/PROTOCOL.md for the labelling rules and provenance guarantees.

Usage:
    python scripts/build_gold.py                       # default: seed=42
    python scripts/build_gold.py --seed 42 --total 600
    make gold
"""

from __future__ import annotations

import argparse
import random
import string
from collections import Counter
from pathlib import Path

from faker import Faker

from forge.schema import HIGH_SEVERITY, PIIRecord, PIISpan, PIIType

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
TARGET_TOTAL = 600
DEV_FRACTION = 0.33

Segment = str | tuple[str, PIIType]
RawTemplate = list[str | PIIType]


# ---------------------------------------------------------------------------
# PII value generator
# ---------------------------------------------------------------------------
class PIIValueGenerator:
    """Produce realistic, diverse PII values using Faker + custom formats."""

    def __init__(self, seed: int):
        self.fake_us = Faker("en_US")
        self.fake_in = Faker("en_IN")
        self.fake_gb = Faker("en_GB")
        self.rng = random.Random(seed)
        Faker.seed(seed)

    def _pick_faker(self) -> Faker:
        return self.rng.choice([self.fake_us, self.fake_in, self.fake_gb])

    def gen(self, pii_type: PIIType) -> str:
        dispatch = {
            PIIType.PERSON: self._person,
            PIIType.EMAIL: self._email,
            PIIType.PHONE: self._phone,
            PIIType.STREET_ADDRESS: self._street_address,
            PIIType.USERNAME: self._username,
            PIIType.URL: self._url,
            PIIType.IP_ADDRESS: self._ip,
            PIIType.LOCATION: self._location,
            PIIType.DATE_OF_BIRTH: self._dob,
            PIIType.AGE: self._age,
            PIIType.CREDIT_CARD: self._credit_card,
            PIIType.BANK_ACCOUNT: self._bank_account,
            PIIType.SSN: self._ssn,
            PIIType.AADHAAR: self._aadhaar,
            PIIType.PAN: self._pan,
            PIIType.PASSPORT: self._passport,
            PIIType.DRIVER_LICENSE: self._driver_license,
            PIIType.PASSWORD: self._password,
            PIIType.API_KEY: self._api_key,
        }
        return dispatch[pii_type]()

    def _person(self) -> str:
        return self._pick_faker().name().replace("\n", " ")

    def _email(self) -> str:
        return self._pick_faker().email()

    def _phone(self) -> str:
        fmt = self.rng.choice([
            "+1 {a}-{b}-{c}",
            "+91 {d} {e}",
            "({a}) {b}-{c}",
            "+44 {f} {g}",
        ])
        return fmt.format(
            a=self._digits(3), b=self._digits(3), c=self._digits(4),
            d=self._digits(5), e=self._digits(5),
            f=self._digits(4), g=self._digits(6),
        )

    def _street_address(self) -> str:
        f = self._pick_faker()
        return f"{f.street_address()}, {f.city()}".replace("\n", ", ")

    def _username(self) -> str:
        return self._pick_faker().user_name()

    def _url(self) -> str:
        return self._pick_faker().url()

    def _ip(self) -> str:
        return self.fake_us.ipv4()

    def _location(self) -> str:
        return self._pick_faker().city().replace("\n", " ")

    def _dob(self) -> str:
        d = self.fake_us.date_of_birth(minimum_age=18, maximum_age=80)
        fmt = self.rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%d %b %Y"])
        return d.strftime(fmt)

    def _age(self) -> str:
        return str(self.rng.randint(18, 85))

    def _credit_card(self) -> str:
        raw = self.fake_us.credit_card_number()
        if len(raw) == 16:
            return f"{raw[:4]} {raw[4:8]} {raw[8:12]} {raw[12:]}"
        return raw

    def _bank_account(self) -> str:
        length = self.rng.choice([8, 10, 12, 14, 16])
        return self._digits(length)

    def _ssn(self) -> str:
        return self.fake_us.ssn()

    def _aadhaar(self) -> str:
        return f"{self._digits(4)} {self._digits(4)} {self._digits(4)}"

    def _pan(self) -> str:
        letters = string.ascii_uppercase
        return (
            self.rng.choice(letters)
            + self.rng.choice(letters)
            + self.rng.choice(letters)
            + self.rng.choice(letters)
            + self.rng.choice(letters)
            + self._digits(4)
            + self.rng.choice(letters)
        )

    def _passport(self) -> str:
        prefix = self.rng.choice(["A", "B", "C", "J", "K", "L", "M", "N", "P", "R", "S", "T"])
        return prefix + self._digits(7)

    def _driver_license(self) -> str:
        state = self.rng.choice(["CA", "NY", "TX", "FL", "IL", "DL", "MH", "KA", "TN"])
        return f"{state}-{self._digits(8)}"

    def _password(self) -> str:
        templates = [
            lambda: f"P@ss{self._digits(3)}!{self._alpha(2)}",
            lambda: f"{self._alpha(4)}{self._digits(2)}#{self._alpha(3)}",
            lambda: f"{self._alpha(3)}_{self._digits(4)}!{self._alpha(2).upper()}",
            lambda: f"Str0ng!{self._alpha(3)}{self._digits(3)}",
            lambda: f"{self._alpha(5).capitalize()}@{self._digits(3)}",
        ]
        return self.rng.choice(templates)()

    def _api_key(self) -> str:
        prefix = self.rng.choice(["sk-proj-", "tok_test_", "FAKEKEY", "key_", "apikey-", "mykey_"])
        length = self.rng.randint(20, 36)
        charset = string.ascii_letters + string.digits
        body = "".join(self.rng.choice(charset) for _ in range(length))
        return prefix + body

    def _digits(self, n: int) -> str:
        return "".join(str(self.rng.randint(0, 9)) for _ in range(n))

    def _alpha(self, n: int) -> str:
        return "".join(self.rng.choice(string.ascii_lowercase) for _ in range(n))


# ---------------------------------------------------------------------------
# Templates — (segments, weight)
# Weight = how many times to instantiate with fresh Faker values.
# Aim: ~600 records total across all templates.
#
# Organized by scenario. Each list entry:
#   (carrier_segments_with_PIIType_slots, instantiation_weight)
# ---------------------------------------------------------------------------
P = PIIType

# fmt: off
TEMPLATES: list[tuple[RawTemplate, int]] = [
    # ===================================================================
    # A. CONTACT / INTRODUCTION  (PERSON, EMAIL, PHONE)
    # ===================================================================
    (["Hi, my name is ", P.PERSON, " and my email is ", P.EMAIL, "."], 6),
    (["Contact ", P.PERSON, " at ", P.PHONE, " for further details."], 6),
    (["You can reach me via email at ", P.EMAIL, " or call ", P.PHONE, "."], 5),
    (["From: ", P.PERSON, " <", P.EMAIL, ">"], 5),
    (["Please forward this to ", P.PERSON, ", their address is ", P.EMAIL, "."], 5),
    (["Name: ", P.PERSON, " | Phone: ", P.PHONE, " | Email: ", P.EMAIL], 6),
    (["My colleague ", P.PERSON, " can be reached at ", P.PHONE, " during business hours."], 5),
    ([P.PERSON, " here. Shoot me a note at ", P.EMAIL, " anytime."], 5),
    (["For inquiries, email ", P.EMAIL, " or reach out to ", P.PERSON, " directly."], 5),
    (["Hi team, adding ", P.PERSON, " (", P.EMAIL, ") to this thread."], 5),

    # ===================================================================
    # B. ADDRESS / LOCATION
    # ===================================================================
    (["Ship to: ", P.STREET_ADDRESS, ", attention ", P.PERSON, "."], 6),
    (["Our new office is located at ", P.STREET_ADDRESS, "."], 5),
    ([P.PERSON, " moved to ", P.LOCATION, " last month."], 5),
    (["Delivery address: ", P.STREET_ADDRESS, ". Contact: ", P.PHONE, "."], 5),
    (["We have branches in ", P.LOCATION, " and ", P.LOCATION, "."], 4),
    (["Return to: ", P.PERSON, ", ", P.STREET_ADDRESS, "."], 5),
    (["The event will be held in ", P.LOCATION, ". RSVP to ", P.EMAIL, "."], 5),
    ([P.PERSON, " currently resides at ", P.STREET_ADDRESS, "."], 5),

    # ===================================================================
    # C. FINANCIAL / PAYMENT  (CREDIT_CARD, BANK_ACCOUNT)
    # ===================================================================
    (["Please charge card ", P.CREDIT_CARD, " for the balance."], 7),
    (["Refund processed to card ", P.CREDIT_CARD, "."], 6),
    (["Wire transfer to account ", P.BANK_ACCOUNT, " completed."], 7),
    (["Invoice paid via card ", P.CREDIT_CARD, " by ", P.PERSON, "."], 6),
    (["Deposit to account ", P.BANK_ACCOUNT, ", beneficiary: ", P.PERSON, "."], 6),
    (["Card ", P.CREDIT_CARD, " was declined. Try another payment method."], 5),
    (["Bank account ", P.BANK_ACCOUNT, " has been linked to your profile."], 5),
    ([P.PERSON, " authorized a charge on card ", P.CREDIT_CARD, " for the subscription."], 5),
    (["Transaction alert: your account ", P.BANK_ACCOUNT, " was debited."], 5),
    (["Please update your card to ", P.CREDIT_CARD, " for auto-renewal."], 5),

    # ===================================================================
    # D. IDENTITY / KYC  (SSN, AADHAAR, PAN, PASSPORT, DRIVER_LICENSE)
    # ===================================================================
    (["SSN on file: ", P.SSN, ". Please verify."], 7),
    (["Aadhaar verification for ", P.PERSON, ": ", P.AADHAAR, "."], 7),
    (["PAN: ", P.PAN, " linked to account holder ", P.PERSON, "."], 7),
    (["Passport number ", P.PASSPORT, " issued to ", P.PERSON, "."], 6),
    (["Driver license ", P.DRIVER_LICENSE, " valid through 2028."], 8),
    (["KYC update: Aadhaar ", P.AADHAAR, ", PAN ", P.PAN, "."], 7),
    (["Your SSN (", P.SSN, ") has been used to apply for credit."], 5),
    (["National ID: ", P.AADHAAR, ". Please keep this confidential."], 6),
    (["Tax filing with PAN ", P.PAN, " for assessment year 2025-26."], 6),
    (["Passport ", P.PASSPORT, " expires on 2029-03-15."], 5),
    (["License ", P.DRIVER_LICENSE, " belongs to ", P.PERSON, "."], 7),
    ([P.PERSON, " submitted Aadhaar ", P.AADHAAR, " for verification."], 6),
    (["Identity proof: DL ", P.DRIVER_LICENSE, " or passport ", P.PASSPORT, "."], 7),
    (["Social security number ", P.SSN, " on the application form."], 5),

    # ===================================================================
    # E. TECHNICAL / IT / SECURITY  (USERNAME, IP, API_KEY, PASSWORD, URL)
    # ===================================================================
    (["Login failed for user ", P.USERNAME, " from ", P.IP_ADDRESS, "."], 7),
    (["API key ", P.API_KEY, " has been rotated. Update your config."], 7),
    (["User ", P.USERNAME, " reset their password to ", P.PASSWORD, "."], 5),
    (["Access denied from IP ", P.IP_ADDRESS, ". Contact admin."], 6),
    (["Profile: ", P.URL, " (user: ", P.USERNAME, ")"], 7),
    (["Token ", P.API_KEY, " expired at 14:32 UTC."], 6),
    (["SSH connection from ", P.IP_ADDRESS, " by ", P.USERNAME, " accepted."], 5),
    (["Your temporary password is ", P.PASSWORD, ". Change it immediately."], 8),
    (["Webhook endpoint: ", P.URL, " with key ", P.API_KEY, "."], 5),
    ([P.USERNAME, " logged in from ", P.IP_ADDRESS, " at 09:15."], 5),
    (["Revoked key ", P.API_KEY, " for user ", P.USERNAME, "."], 5),
    (["Visit ", P.URL, " to reset the credentials."], 7),
    (["Failed password attempt: '", P.PASSWORD, "' for ", P.USERNAME, "."], 4),
    (["Firewall blocked ", P.IP_ADDRESS, " after 5 failed attempts."], 5),

    # ===================================================================
    # F. PERSONAL / DEMOGRAPHIC  (DATE_OF_BIRTH, AGE, PERSON)
    # ===================================================================
    ([P.PERSON, ", born ", P.DATE_OF_BIRTH, ", has been enrolled."], 6),
    (["Patient ", P.PERSON, ", age ", P.AGE, ", was seen today."], 6),
    (["DOB: ", P.DATE_OF_BIRTH, ". Applicant: ", P.PERSON, "."], 5),
    (["Subscriber ", P.PERSON, " is ", P.AGE, " years old."], 5),
    (["Birthday: ", P.DATE_OF_BIRTH, ". Send a card to ", P.PERSON, "!"], 4),
    (["Age: ", P.AGE, ". Date of birth: ", P.DATE_OF_BIRTH, "."], 5),
    ([P.PERSON, " (", P.AGE, ") requested an account upgrade."], 5),
    (["Records show ", P.PERSON, " was born on ", P.DATE_OF_BIRTH, " in ", P.LOCATION, "."], 5),

    # ===================================================================
    # G. COMMUNICATION / MESSAGING
    # ===================================================================
    (["Hey ", P.PERSON, ", just emailed you at ", P.EMAIL, ". Check it out!"], 5),
    (["Call me at ", P.PHONE, " when you get a chance. — ", P.PERSON], 5),
    ([P.PERSON, " shared a link: ", P.URL, ". Worth reading."], 7),
    (["Forwarding this to ", P.EMAIL, " per ", P.PERSON, "'s request."], 4),
    (["Reminder: call ", P.PERSON, " at ", P.PHONE, " before 5pm."], 4),
    (["Text ", P.PHONE, " or DM ", P.USERNAME, " for the invite code."], 4),
    (["Meeting notes sent to ", P.EMAIL, ". Let me know, ", P.PERSON, "."], 4),

    # ===================================================================
    # H. MIXED / COMPLEX  (3+ PII types per record)
    # ===================================================================
    (["Customer: ", P.PERSON, ", email: ", P.EMAIL, ", card: ", P.CREDIT_CARD, "."], 6),
    ([P.PERSON, " (SSN: ", P.SSN, ") at ", P.STREET_ADDRESS, ", phone ", P.PHONE, "."], 5),
    (["New user: ", P.USERNAME, " (", P.EMAIL, ") from ", P.IP_ADDRESS, "."], 5),
    (["Ticket #4521: ", P.PERSON, ", DOB ", P.DATE_OF_BIRTH, ", Aadhaar ", P.AADHAAR, ", PAN ", P.PAN, "."], 6),
    (["Account holder ", P.PERSON, " with bank account ", P.BANK_ACCOUNT, " and card ", P.CREDIT_CARD, "."], 5),
    (["Onboarding: ", P.PERSON, ", ", P.EMAIL, ", passport ", P.PASSPORT, ", address ", P.STREET_ADDRESS, "."], 5),
    ([P.PERSON, " (", P.AGE, ", ", P.LOCATION, ") applied with PAN ", P.PAN, "."], 5),
    (["Support case: user ", P.USERNAME, " (", P.EMAIL, ") from ", P.IP_ADDRESS, " reports password '", P.PASSWORD, "' not working."], 4),
    (["Payroll: ", P.PERSON, ", SSN ", P.SSN, ", account ", P.BANK_ACCOUNT, ", born ", P.DATE_OF_BIRTH, "."], 5),
    (["Shipping ", P.PERSON, " at ", P.STREET_ADDRESS, " (card ", P.CREDIT_CARD, ", phone ", P.PHONE, ")."], 5),
    (["Booking for ", P.PERSON, ", passport ", P.PASSPORT, ", DOB ", P.DATE_OF_BIRTH, ", email ", P.EMAIL, "."], 5),
    ([P.PERSON, ", DL ", P.DRIVER_LICENSE, ", address ", P.STREET_ADDRESS, ", phone ", P.PHONE, "."], 5),

    # ===================================================================
    # I. SINGLE-ENTITY RECORDS  (1 PII type — tests precision)
    # ===================================================================
    (["The email on file is ", P.EMAIL, "."], 4),
    (["Card number: ", P.CREDIT_CARD, "."], 5),
    (["Aadhaar: ", P.AADHAAR, "."], 5),
    (["PAN number: ", P.PAN, "."], 5),
    (["SSN: ", P.SSN, "."], 5),
    (["Account: ", P.BANK_ACCOUNT, "."], 5),
    (["API key: ", P.API_KEY, "."], 5),
    (["Password: ", P.PASSWORD, "."], 6),
    ([P.PERSON, " signed the document."], 4),
    (["Delivered to ", P.STREET_ADDRESS, "."], 4),

    # ===================================================================
    # J. EDGE CASES  (PII at boundaries, back-to-back, multi-sentence)
    # ===================================================================
    ([P.PERSON, " is the account holder."], 4),
    (["Contact details: ", P.EMAIL, " ", P.PHONE, "."], 4),
    (["Thank you, ", P.PERSON, ". Your order ships to ", P.STREET_ADDRESS, ". Your card ", P.CREDIT_CARD, " was charged."], 4),
    (["Refer to the page at ", P.URL, " for more information. Questions? Ask ", P.PERSON, "."], 4),
    (["Update: user ", P.USERNAME, " changed their API key. New key: ", P.API_KEY, ". Acknowledge this change."], 4),
    (["Aadhaar ", P.AADHAAR, " and PAN ", P.PAN, " and passport ", P.PASSPORT, " are linked."], 5),
    (["Alert: login from ", P.IP_ADDRESS, " for ", P.USERNAME, " with token ", P.API_KEY, " at 03:42 UTC. Suspicious activity detected on account ", P.BANK_ACCOUNT, "."], 4),

    # ===================================================================
    # K. NEGATIVE EXAMPLES  (no PII — tests false-positive rate)
    # ===================================================================
    (["The weather in Mumbai is pleasant today and trains are on time."], 5),
    (["Our quarterly revenue grew 12% compared to last year."], 5),
    (["Please review the attached document and share your feedback."], 5),
    (["The server migration is scheduled for next weekend."], 4),
    (["Thank you for your purchase. Your order will arrive in 3-5 business days."], 4),
    (["Meeting rescheduled to Thursday at 2pm in Conference Room B."], 4),
    (["The new policy takes effect on January 1st across all departments."], 4),
    (["System maintenance window: Saturday 2am to 6am IST."], 4),
    (["All tests passed. Build #1247 is ready for staging."], 3),
    (["No issues reported during the last deployment cycle."], 3),
]
# fmt: on


# ---------------------------------------------------------------------------
# Record builder (shared with make_sample_gold.py logic)
# ---------------------------------------------------------------------------
def build_record(
    record_id: str,
    segments: list[Segment],
    split: str,
    lang: str = "en",
) -> PIIRecord:
    text_parts: list[str] = []
    spans: list[PIISpan] = []
    cursor = 0
    for seg in segments:
        if isinstance(seg, str):
            text_parts.append(seg)
            cursor += len(seg)
        else:
            value, label = seg
            start = cursor
            end = start + len(value)
            spans.append(PIISpan(start=start, end=end, label=label, text=value))
            text_parts.append(value)
            cursor = end
    return PIIRecord(
        id=record_id,
        text="".join(text_parts),
        spans=spans,
        lang=lang,
        source="synthetic:faker",
        split=split,
    )


def expand_template(tpl: RawTemplate, gen: PIIValueGenerator) -> list[Segment]:
    segments: list[Segment] = []
    for part in tpl:
        if isinstance(part, PIIType):
            segments.append((gen.gen(part), part))
        else:
            segments.append(part)
    return segments


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def print_stats(records: list[PIIRecord], label: str) -> None:
    type_counts: Counter[str] = Counter()
    records_with_pii = 0
    negative_count = 0
    for r in records:
        if r.spans:
            records_with_pii += 1
            for s in r.spans:
                type_counts[s.label.value] += 1
        else:
            negative_count += 1

    print(f"\n{'=' * 60}")
    print(f"  {label}: {len(records)} records ({records_with_pii} positive, {negative_count} negative)")
    print(f"{'=' * 60}")
    print(f"  {'PII Type':<20} {'Count':>6}  {'High-sev':>8}")
    print(f"  {'-' * 38}")
    for t in PIIType:
        count = type_counts.get(t.value, 0)
        sev = " !!!" if t in HIGH_SEVERITY else ""
        warn = " [LOW]" if count < 15 else ""
        print(f"  {t.value:<20} {count:>6}{sev}{warn}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold dev/test sets.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: data/gold/ in project root).",
    )
    args = parser.parse_args()

    gen = PIIValueGenerator(args.seed)
    rng = random.Random(args.seed + 1)

    all_records: list[PIIRecord] = []
    seq = 0
    for tpl, weight in TEMPLATES:
        for _ in range(weight):
            seq += 1
            segments = expand_template(tpl, gen)
            record = build_record(f"pii-{seq:04d}", segments, split="dev")
            all_records.append(record)

    rng.shuffle(all_records)

    n_dev = int(len(all_records) * DEV_FRACTION)
    dev_records = all_records[:n_dev]
    test_records = all_records[n_dev:]

    for i, r in enumerate(dev_records, 1):
        dev_records[i - 1] = r.model_copy(update={"id": f"pii-dev-{i:04d}", "split": "dev"})
    for i, r in enumerate(test_records, 1):
        test_records[i - 1] = r.model_copy(update={"id": f"pii-test-{i:04d}", "split": "test"})

    out_dir = args.output_dir or Path(__file__).resolve().parents[1] / "data" / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_path = out_dir / "dev.jsonl"
    test_path = out_dir / "test.jsonl"

    for path, records in [(dev_path, dev_records), (test_path, test_records)]:
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")

    print(f"wrote {len(dev_records)} records -> {dev_path}")
    print(f"wrote {len(test_records)} records -> {test_path}")
    print_stats(dev_records, "dev")
    print_stats(test_records, "test")
    print_stats(dev_records + test_records, "TOTAL (dev + test)")

    print("demo redaction (first test record):")
    print(f"  text:     {test_records[0].text}")
    print(f"  redacted: {test_records[0].redacted}")


if __name__ == "__main__":
    main()
