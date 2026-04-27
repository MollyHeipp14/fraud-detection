import pandas as pd
from features import build_model_frame


def _make_inputs(amount_usd, failed_logins_24h):
    transactions = pd.DataFrame([{
        "transaction_id": 1,
        "account_id": 99,
        "amount_usd": amount_usd,
        "failed_logins_24h": failed_logins_24h,
    }])
    accounts = pd.DataFrame([{"account_id": 99}])
    return transactions, accounts


# ---------------------------------------------------------------------------
# is_large_amount  (>= 1000 → 1, else → 0)
# ---------------------------------------------------------------------------

def test_is_large_amount_at_threshold():
    txs, accts = _make_inputs(amount_usd=1000, failed_logins_24h=0)
    df = build_model_frame(txs, accts)
    assert df["is_large_amount"].iloc[0] == 1


def test_is_large_amount_above_threshold():
    txs, accts = _make_inputs(amount_usd=2500, failed_logins_24h=0)
    df = build_model_frame(txs, accts)
    assert df["is_large_amount"].iloc[0] == 1


def test_is_large_amount_below_threshold():
    txs, accts = _make_inputs(amount_usd=999, failed_logins_24h=0)
    df = build_model_frame(txs, accts)
    assert df["is_large_amount"].iloc[0] == 0


# ---------------------------------------------------------------------------
# login_pressure  bins: (-1,0] → none, (0,2] → low, (2,100] → high
# ---------------------------------------------------------------------------

def test_login_pressure_none():
    txs, accts = _make_inputs(amount_usd=100, failed_logins_24h=0)
    df = build_model_frame(txs, accts)
    assert df["login_pressure"].iloc[0] == "none"


def test_login_pressure_low_at_boundary():
    txs, accts = _make_inputs(amount_usd=100, failed_logins_24h=1)
    df = build_model_frame(txs, accts)
    assert df["login_pressure"].iloc[0] == "low"


def test_login_pressure_low_at_upper_boundary():
    txs, accts = _make_inputs(amount_usd=100, failed_logins_24h=2)
    df = build_model_frame(txs, accts)
    assert df["login_pressure"].iloc[0] == "low"


def test_login_pressure_high():
    txs, accts = _make_inputs(amount_usd=100, failed_logins_24h=3)
    df = build_model_frame(txs, accts)
    assert df["login_pressure"].iloc[0] == "high"


def test_login_pressure_high_extreme():
    txs, accts = _make_inputs(amount_usd=100, failed_logins_24h=10)
    df = build_model_frame(txs, accts)
    assert df["login_pressure"].iloc[0] == "high"


# ---------------------------------------------------------------------------
# Account merge
# ---------------------------------------------------------------------------

def test_account_fields_present_after_merge():
    transactions = pd.DataFrame([{"transaction_id": 1, "account_id": 99, "amount_usd": 500, "failed_logins_24h": 0}])
    accounts = pd.DataFrame([{"account_id": 99, "prior_chargebacks": 2, "kyc_level": "full"}])
    df = build_model_frame(transactions, accounts)
    assert df["prior_chargebacks"].iloc[0] == 2
    assert df["kyc_level"].iloc[0] == "full"
