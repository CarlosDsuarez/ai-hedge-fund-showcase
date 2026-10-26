"""Position sizing for a single name asked for at the terminal.

main.py sizes a BATCH (main.py, "Equal-weight sizing"): deployable_fraction
divided by num_buys, scaled by the vol regime, clamped to [MIN_POSITION_PCT,
MAX_POSITION_PCT]. Feed that formula one ticker and num_buys becomes 1, so the
raw allocation is the WHOLE deployable fraction — about 1.4x equity on the
current buying-power budget — and MAX_POSITION_PCT stops being a ceiling and
becomes the answer. Every single-name proposal would come out at exactly 15%
of equity, whatever the name.

The denominator therefore gets a floor taken from config's own arithmetic:

    n_floor = ceil(deployable_frac / MAX_POSITION_PCT)

is the number of names it takes to spend the budget at the cap. Sizing this
trade as one of those keeps the clamp a ceiling again, and peer_count raises
the denominator further when today's run actually found more candidates.

The batch also gets three limits implicitly that an ad-hoc single trade does
not: the gross-exposure cap (inside compute_deployment_budget), the per-position
cap (main.py's weight_blocked_tickers only blocks names ALREADY over it — it
never checks whether a new buy would cross it), and the sector cap. All three
are applied here, and reported as (limit, headroom) in dollars, which is what
"which gate fails and by how much" actually means to someone deciding.
"""
import math
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from agents.budget import compute_deployment_budget, deployable_fraction


@dataclass(frozen=True)
class Cap:
    """One dollar limit on this trade, and how much room is left under it."""
    name: str
    label: str
    limit: float
    headroom: float

    @property
    def blocked(self) -> bool:
        return self.headroom <= 0


@dataclass
class SizeProposal:
    ticker: str
    price: float
    qty: int = 0
    spend: float = 0.0
    target_dollars: float = 0.0
    alloc_pct: float = 0.0          # fraction of equity, pre-cap
    n: int = 1
    n_floor: int = 1
    peer_count: int = 1
    deployable: float = 0.0
    deployable_frac: float = 0.0
    vol_size_mult: float = 1.0
    budget_degraded: bool = False
    budget_reason: str = ""
    caps: list = field(default_factory=list)
    binding: str = ""               # name of the cap that set the size

    def cap(self, name: str):
        return next((c for c in self.caps if c.name == name), None)


def peer_count_from_run(run: dict) -> int:
    """How many names cleared today's gates, from the cached run blob.

    The funnel records one row per ticker and stamps a per-stage count, so
    stages["gates"] is exactly the batch main.py would have divided the budget
    across. Falling back to the executed+queued list keeps this working on runs
    written before the stage counters existed.
    """
    if not run:
        return 1
    funnel = run.get("funnel") or {}
    stages = funnel.get("stages") or {}
    gates = stages.get("gates")
    if gates:
        return max(int(gates), 1)
    survivors = len(funnel.get("executed") or []) + len(funnel.get("queued") or [])
    if survivors:
        return survivors
    return max(len(run.get("orders_submitted") or []), 1)


def suggest_size(*, ticker: str, price: float, equity: float,
                 buying_power: float, long_market_value: float = 0.0,
                 vol_size_mult: float = 1.0, peer_count: int = 1,
                 current_mv: float = 0.0, sector_mv: float = 0.0) -> SizeProposal:
    """How many shares this name should get, and what stopped it being more."""
    equity = max(float(equity or 0.0), 0.0)
    price = float(price or 0.0)
    proposal = SizeProposal(ticker=ticker, price=price, peer_count=max(peer_count, 1),
                            vol_size_mult=vol_size_mult)
    if equity <= 0 or price <= 0:
        proposal.budget_reason = "sin equity o sin precio — no se puede dimensionar"
        return proposal

    budget = compute_deployment_budget(
        equity, buying_power=buying_power, long_market_value=long_market_value)
    frac = deployable_fraction(budget["deployable"], equity)

    proposal.deployable = budget["deployable"]
    proposal.deployable_frac = frac
    proposal.budget_degraded = bool(budget["degraded"])
    proposal.budget_reason = budget["reason"]

    # The denominator: never 1 while the budget could fund more names than one.
    n_floor = max(math.ceil(frac / config.MAX_POSITION_PCT), 1) if frac > 0 else 1
    n = max(proposal.peer_count, n_floor, 1)
    proposal.n_floor = n_floor
    proposal.n = n

    alloc = min(frac / n * vol_size_mult, config.MAX_POSITION_PCT)
    alloc = max(alloc, config.MIN_POSITION_PCT)
    proposal.alloc_pct = alloc
    proposal.target_dollars = equity * alloc

    # Sector cap scaled by the deployable fraction, exactly as main.py does it:
    # as a flat share of equity it would bite immediately under leverage and
    # block the very deployment the budget just authorised.
    sector_limit = config.MAX_SECTOR_WEIGHT * max(frac, 1.0) * equity
    position_limit = config.MAX_POSITION_PCT * equity

    proposal.caps = [
        Cap("target", f"objetivo {alloc:.1%} del equity",
            proposal.target_dollars, proposal.target_dollars),
        Cap("budget", f"presupuesto desplegable ({config.BUDGET_ALLOCATION:.0%} de {budget['base']})",
            budget["deployable"], budget["deployable"]),
        Cap("position", f"máx. por posición {config.MAX_POSITION_PCT:.0%} del equity",
            position_limit, position_limit - current_mv),
        Cap("sector", f"máx. por sector {config.MAX_SECTOR_WEIGHT:.0%} del libro",
            sector_limit, sector_limit - sector_mv),
        Cap("buying_power", "buying power del bróker",
            float(buying_power or 0.0), float(buying_power or 0.0)),
    ]

    binding = min(proposal.caps, key=lambda c: c.headroom)
    spend = max(min(c.headroom for c in proposal.caps), 0.0)
    proposal.binding = binding.name
    proposal.spend = spend
    proposal.qty = math.floor(spend / price) if price > 0 else 0
    return proposal


def min_ticket(equity: float) -> float:
    """The smallest position the pipeline will fund — main.py's BELOW_MIN_TICKET line."""
    return config.MIN_POSITION_PCT * max(float(equity or 0.0), 0.0)
