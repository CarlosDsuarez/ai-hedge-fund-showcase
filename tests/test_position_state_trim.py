"""execution/order_manager.reduce_position_qty — keep the record in sync with reality.

A sell placed outside the exit engine (a chat trim) must shrink the tracked
quantity: qty_remaining is what the ledger records for an untagged exit and
what a stateless remainder would otherwise be re-registered at, and a record
claiming shares the broker no longer holds is exactly the phantom holding
the reconciliation exists to catch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import execution.order_manager as om


def _record(qty=100):
    return {
        "entry_price": 100.0, "atr_14": 2.0,
        "initial_stop": 95.0, "trail_stop": 97.0, "highest_close": 103.0,
        "qty_total": qty, "qty_remaining": qty, "decay_count": 2,
        "entry_time": "2026-09-01T10:00:00",
    }


def _seed(record):
    om.save_position_state({"AAPL": record})


def test_trim_reduces_both_counters():
    _seed(_record(100))
    updated = om.reduce_position_qty("AAPL", 40)
    on_disk = om.load_position_state()["AAPL"]

    assert updated["qty_remaining"] == 60
    assert updated["qty_total"] == 60
    assert on_disk["qty_total"] == 60


def test_trim_to_zero_deletes_the_record():
    """A flat name must not linger as a phantom holding."""
    _seed(_record(10))
    assert om.reduce_position_qty("AAPL", 10) is None
    assert om.load_position_state() == {}


def test_overselling_also_deletes_the_record():
    _seed(_record(10))
    assert om.reduce_position_qty("AAPL", 25) is None
    assert om.load_position_state() == {}


def test_untracked_ticker_is_a_no_op():
    _seed(_record(10))
    assert om.reduce_position_qty("MSFT", 5) is None
    assert "AAPL" in om.load_position_state()


def test_stops_and_counters_survive_the_trim():
    """Only the quantities change: the stops and the decay counter stay put."""
    _seed(_record(100))
    updated = om.reduce_position_qty("AAPL", 30)

    assert updated["initial_stop"] == 95.0
    assert updated["trail_stop"] == 97.0
    assert updated["highest_close"] == 103.0
    assert updated["decay_count"] == 2
