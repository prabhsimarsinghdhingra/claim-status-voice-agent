import pytest
from hypothesis import given
from hypothesis import strategies as st

from csva.models import ClaimStatusResult
from csva.parsing import digits_from_text, ground, parse_claim_status, reconcile


class TestDigits:
    def test_spoken_and_numeric(self):
        assert digits_from_text("one two three oh 4 five") == "123045"

    def test_dates_and_punctuation(self):
        assert digits_from_text("04/12/1985") == "04121985"
        assert digits_from_text("NPI: 12-34") == "1234"

    def test_ignores_noise_words(self):
        assert digits_from_text("hello there") == ""

    @given(st.text())
    def test_never_raises_and_only_digits(self, s):
        assert digits_from_text(s).isdigit() or digits_from_text(s) == ""


class TestParseClaimStatus:
    def test_menu_text_is_not_a_result(self):
        assert parse_claim_status("For claim status, press 2. To speak with a representative, press 0.") is None

    def test_post_result_menu_is_not_a_result(self):
        assert parse_claim_status("To hear that again, press 1. To end this call, press 9.") is None

    def test_denied_ivr(self):
        r = parse_claim_status(
            "Status: denied. Denial code CO-50: Not medically necessary. Billed amount $310.00. "
            "Paid amount $0.00. You may file an appeal within 180 days. Call reference number REF-12345."
        )
        assert r is not None
        assert (r.status, r.denial_code, r.next_action, r.reference_number) == (
            "denied",
            "CO-50",
            "appeal",
            "REF-12345",
        )
        assert (r.billed_amount, r.paid_amount) == (310.0, 0.0)
        assert r.denial_reason == "Not medically necessary"

    def test_denied_missing_info_means_resubmit(self):
        r = parse_claim_status(
            "Status: denied. Denial code CO-16: Claim lacks information. Please submit a corrected claim."
        )
        assert r is not None and r.next_action == "resubmit"

    def test_paid_with_trace_and_thousands_separator(self):
        r = parse_claim_status("Status: paid. Billed amount $1,210.00. Paid amount $1,142.50. trace number EFT88231.")
        assert r is not None
        assert (r.status, r.billed_amount, r.paid_amount, r.check_number) == ("paid", 1210.0, 1142.5, "EFT88231")

    def test_pending_means_follow_up(self):
        r = parse_claim_status("Status: pending. This claim is still being processed.")
        assert r is not None and (r.status, r.next_action) == ("pending", "follow_up")

    def test_rep_phrasing(self):
        r = parse_claim_status(
            "Okay, I have claim 1 here. It was denied. The denial code is CO-50: Not necessary. "
            "The billed amount was $310.00 and the paid amount is $0.00."
        )
        assert r is not None and (r.status, r.denial_code, r.billed_amount) == ("denied", "CO-50", 310.0)

    @pytest.mark.parametrize(
        "text", ["We could not find a claim with that number.", "I'm sorry, I could not find a claim."]
    )
    def test_not_found(self, text):
        r = parse_claim_status(text)
        assert r is not None and (r.status, r.next_action) == ("not_found", "verify_claim_number")

    def test_lowercase_denial_code_is_normalised(self):
        r = parse_claim_status("Status: denied. denial code co 50, not covered.")
        assert r is not None and r.denial_code == "CO-50"

    @given(st.text(max_size=400))
    def test_never_raises_on_arbitrary_text(self, s):
        parse_claim_status(s)


class TestGuardrails:
    PAYER = "Status: denied. Denial code CO-50: Not necessary. Billed amount $310.00. Paid amount $0.00. Call reference number REF-25838."

    def test_grounded_values_survive(self):
        r = ClaimStatusResult(
            status="denied", denial_code="CO-50", billed_amount=310.0, paid_amount=0.0, reference_number="REF-25838"
        )
        out, dropped = ground(r, self.PAYER)
        assert dropped == [] and out == r

    def test_hallucinated_values_are_dropped(self):
        r = ClaimStatusResult(
            status="denied",
            denial_code="CO-97",
            billed_amount=999.99,
            paid_amount=12.0,
            reference_number="REF-00000",
            check_number="EFT1234567",
        )
        out, dropped = ground(r, self.PAYER)
        assert set(dropped) == {"denial_code", "billed_amount", "paid_amount", "reference_number", "check_number"}
        assert out.denial_code is None and out.billed_amount is None and out.status == "denied"

    def test_spoken_code_with_space_still_grounded(self):
        r = ClaimStatusResult(status="denied", denial_code="CO-50")
        out, dropped = ground(r, "the denial code is c o 50")
        assert dropped == [] and out.denial_code == "CO-50"

    def test_integer_amount_matches_either_format(self):
        r = ClaimStatusResult(status="paid", paid_amount=142.0)
        assert ground(r, "paid amount $142.00")[1] == []
        assert ground(r, "paid amount 142 dollars")[1] == []

    def test_reconcile_parser_wins_on_conflict_but_model_fills_gaps(self):
        model = ClaimStatusResult(status="paid", paid_amount=5.0, check_number="EFT99999")
        parsed = ClaimStatusResult(status="denied", paid_amount=0.0, next_action="appeal")
        merged = reconcile(model, parsed)
        assert (merged.status, merged.paid_amount, merged.next_action) == ("denied", 0.0, "appeal")
        assert merged.check_number == "EFT99999"  # parser had nothing -> model value kept (then grounded separately)
