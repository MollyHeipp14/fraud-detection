import pandas as pd
from analyze_fraud import summarize_results


def _scored(*rows):
    return pd.DataFrame(rows)


def _chargebacks(*transaction_ids):
    return pd.DataFrame({"transaction_id": list(transaction_ids)})


# ---------------------------------------------------------------------------
# chargeback_rate
# ---------------------------------------------------------------------------

def test_chargeback_rate_is_1_when_all_are_fraud():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 500},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 300},
    )
    summary = summarize_results(scored, _chargebacks(1, 2))
    row = summary[summary["risk_label"] == "high"].iloc[0]
    assert row["chargebacks"] == 2
    assert row["chargeback_rate"] == 1.0


def test_chargeback_rate_is_0_when_no_fraud():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "low", "amount_usd": 50},
        {"transaction_id": 2, "risk_label": "low", "amount_usd": 30},
    )
    summary = summarize_results(scored, _chargebacks())
    row = summary[summary["risk_label"] == "low"].iloc[0]
    assert row["chargebacks"] == 0
    assert row["chargeback_rate"] == 0.0


def test_chargeback_rate_partial():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "medium", "amount_usd": 200},
        {"transaction_id": 2, "risk_label": "medium", "amount_usd": 400},
        {"transaction_id": 3, "risk_label": "medium", "amount_usd": 300},
        {"transaction_id": 4, "risk_label": "medium", "amount_usd": 100},
    )
    summary = summarize_results(scored, _chargebacks(1, 3))
    row = summary[summary["risk_label"] == "medium"].iloc[0]
    assert row["chargebacks"] == 2
    assert row["chargeback_rate"] == 0.5


def test_chargebacks_counted_per_label():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 1000},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 800},
        {"transaction_id": 3, "risk_label": "low",  "amount_usd": 50},
        {"transaction_id": 4, "risk_label": "low",  "amount_usd": 30},
    )
    summary = summarize_results(scored, _chargebacks(1))
    high = summary[summary["risk_label"] == "high"].iloc[0]
    low  = summary[summary["risk_label"] == "low"].iloc[0]
    assert high["chargebacks"] == 1
    assert high["chargeback_rate"] == 0.5
    assert low["chargebacks"] == 0
    assert low["chargeback_rate"] == 0.0


# ---------------------------------------------------------------------------
# Transaction counts
# ---------------------------------------------------------------------------

def test_transaction_count_per_label():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "low",  "amount_usd": 10},
        {"transaction_id": 2, "risk_label": "low",  "amount_usd": 20},
        {"transaction_id": 3, "risk_label": "high", "amount_usd": 500},
    )
    summary = summarize_results(scored, _chargebacks())
    assert summary[summary["risk_label"] == "low"]["transactions"].iloc[0] == 2
    assert summary[summary["risk_label"] == "high"]["transactions"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Amount metrics
# ---------------------------------------------------------------------------

def test_total_and_avg_amount_per_label():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "low", "amount_usd": 100},
        {"transaction_id": 2, "risk_label": "low", "amount_usd": 300},
    )
    summary = summarize_results(scored, _chargebacks())
    row = summary[summary["risk_label"] == "low"].iloc[0]
    assert row["total_amount_usd"] == 400
    assert row["avg_amount_usd"] == 200


def test_amounts_are_independent_across_labels():
    scored = _scored(
        {"transaction_id": 1, "risk_label": "low",  "amount_usd": 50},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 1000},
    )
    summary = summarize_results(scored, _chargebacks())
    low  = summary[summary["risk_label"] == "low"].iloc[0]
    high = summary[summary["risk_label"] == "high"].iloc[0]
    assert low["total_amount_usd"] == 50
    assert high["total_amount_usd"] == 1000
