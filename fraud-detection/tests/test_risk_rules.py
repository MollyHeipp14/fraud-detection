from risk_rules import label_risk, score_transaction

# Neutral baseline: every field below every threshold, so score == 0.
# Override one field at a time to test each signal in isolation.
BASE_TX = {
    "device_risk_score": 5,
    "is_international": 0,
    "amount_usd": 100,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
}


def _tx(**overrides):
    return {**BASE_TX, **overrides}


# ---------------------------------------------------------------------------
# label_risk
# ---------------------------------------------------------------------------

def test_label_risk_boundaries():
    assert label_risk(0) == "low"
    assert label_risk(29) == "low"
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"
    assert label_risk(60) == "high"
    assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# device_risk_score  (high >= 70 → +25, medium 40–69 → +10, low < 40 → 0)
# ---------------------------------------------------------------------------

def test_high_device_risk_adds_25():
    assert score_transaction(_tx(device_risk_score=70)) == 25
    assert score_transaction(_tx(device_risk_score=95)) == 25


def test_medium_device_risk_adds_10():
    assert score_transaction(_tx(device_risk_score=40)) == 10
    assert score_transaction(_tx(device_risk_score=69)) == 10


def test_low_device_risk_adds_nothing():
    assert score_transaction(_tx(device_risk_score=39)) == 0


# ---------------------------------------------------------------------------
# is_international  (1 → +15, 0 → 0)
# ---------------------------------------------------------------------------

def test_international_adds_15():
    assert score_transaction(_tx(is_international=1)) == 15


def test_domestic_adds_nothing():
    assert score_transaction(_tx(is_international=0)) == 0


# ---------------------------------------------------------------------------
# amount_usd  (>= 1000 → +25, >= 500 → +10, < 500 → 0)
# ---------------------------------------------------------------------------

def test_large_amount_adds_25():
    assert score_transaction(_tx(amount_usd=1000)) == 25
    assert score_transaction(_tx(amount_usd=5000)) == 25


def test_medium_amount_adds_10():
    assert score_transaction(_tx(amount_usd=500)) == 10
    assert score_transaction(_tx(amount_usd=999)) == 10


def test_small_amount_adds_nothing():
    assert score_transaction(_tx(amount_usd=499)) == 0


# ---------------------------------------------------------------------------
# velocity_24h  (>= 6 → +20, >= 3 → +5, < 3 → 0)
# ---------------------------------------------------------------------------

def test_high_velocity_adds_20():
    assert score_transaction(_tx(velocity_24h=6)) == 20
    assert score_transaction(_tx(velocity_24h=10)) == 20


def test_medium_velocity_adds_5():
    assert score_transaction(_tx(velocity_24h=3)) == 5
    assert score_transaction(_tx(velocity_24h=5)) == 5


def test_low_velocity_adds_nothing():
    assert score_transaction(_tx(velocity_24h=2)) == 0


# ---------------------------------------------------------------------------
# failed_logins_24h  (>= 5 → +20, >= 2 → +10, < 2 → 0)
# ---------------------------------------------------------------------------

def test_high_failed_logins_adds_20():
    assert score_transaction(_tx(failed_logins_24h=5)) == 20
    assert score_transaction(_tx(failed_logins_24h=9)) == 20


def test_medium_failed_logins_adds_10():
    assert score_transaction(_tx(failed_logins_24h=2)) == 10
    assert score_transaction(_tx(failed_logins_24h=4)) == 10


def test_no_failed_logins_adds_nothing():
    assert score_transaction(_tx(failed_logins_24h=0)) == 0


# ---------------------------------------------------------------------------
# prior_chargebacks  (>= 2 → +20, == 1 → +5, 0 → 0)
# ---------------------------------------------------------------------------

def test_multiple_prior_chargebacks_adds_20():
    assert score_transaction(_tx(prior_chargebacks=2)) == 20
    assert score_transaction(_tx(prior_chargebacks=5)) == 20


def test_one_prior_chargeback_adds_5():
    assert score_transaction(_tx(prior_chargebacks=1)) == 5


def test_no_prior_chargebacks_adds_nothing():
    assert score_transaction(_tx(prior_chargebacks=0)) == 0


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------

def test_score_clamped_at_100():
    tx = {
        "device_risk_score": 90,
        "is_international": 1,
        "amount_usd": 5000,
        "velocity_24h": 10,
        "failed_logins_24h": 8,
        "prior_chargebacks": 3,
    }
    assert score_transaction(tx) == 100


def test_score_minimum_is_zero():
    assert score_transaction(BASE_TX) == 0


# ---------------------------------------------------------------------------
# Integration: real-data profiles
# ---------------------------------------------------------------------------

def test_known_fraud_profile_scores_high():
    # Mirrors transaction 50011 from the dataset (confirmed chargeback, $1,400)
    tx = {
        "device_risk_score": 85,
        "is_international": 1,
        "amount_usd": 1400,
        "velocity_24h": 8,
        "failed_logins_24h": 7,
        "prior_chargebacks": 1,
    }
    score = score_transaction(tx)
    assert score >= 60
    assert label_risk(score) == "high"


def test_clean_profile_scores_low():
    # Mirrors transaction 50001 from the dataset (no chargeback, $45)
    tx = {
        "device_risk_score": 8,
        "is_international": 0,
        "amount_usd": 45,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    score = score_transaction(tx)
    assert score < 30
    assert label_risk(score) == "low"
