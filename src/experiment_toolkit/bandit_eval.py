"""
Measuring the lift of a creative that a bandit picked from many generated
variants.

Generative tools make it cheap to produce dozens of creative variants
(thumbnails, headlines, ad copy). A common setup keeps a fixed share of
traffic on the incumbent creative (the control) and lets a bandit move the
rest toward whichever variant is ahead. When the test ends, the team reports
the leader's lift. This module asks how far that number is from the truth and
which estimates from the logged decisions can be trusted.

Two separate biases
-------------------
1. **Adaptive allocation confounds raw arm means with time.** A variant that
   wins traffic late is measured mostly on late traffic. If the click rate
   drifts over the test (a weekend, a promotion, seasonality), its raw mean is
   compared with a control measured evenly over the whole window.
   Inverse-propensity estimators (IPS, self-normalised IPS, doubly robust)
   that use the logged allocation probabilities remove this for a *fixed*
   target arm.
2. **The winner is chosen because its estimate is high** (the winner's curse).
   This is a selection effect. It survives every estimator applied to the
   data that chose the winner, IPS included. Only data that played no part in
   the choice removes it: a fresh confirmation test, or an *evaluation slice*
   of impressions that the bandit serves but whose outcomes it never learns
   from. Splitting an ordinary bandit log at random after the fact is not
   enough: the bandit allocated traffic using both halves, so the "held-out"
   half has already shaped which variant looks best in the other half.

Estimand
--------
For arm ``k`` the value is the mean click rate if every impression in the
evaluated window had been shown ``k``. The lift is ``value(k) - value(control)``.

Log format
----------
One row per impression: the arm shown (``arms``), the outcome (``rewards``),
and the probability with which the logging policy chose that arm
(``propensities``). For a doubly robust estimate, also a per-impression
prediction of the target arm's outcome that was fixed before the impression
was served (for example the bandit's own posterior mean at decision time).

Scope
-----
This module evaluates logged bandit traffic. It is not a bandit framework:
:func:`simulate_bandit_log` is a minimal data generator with known ground
truth, used to test the estimators.

References: Horvitz & Thompson (1952); Swaminathan & Joachims (2015) for
SNIPS; Dudik, Langford & Li (2011) for doubly robust policy evaluation;
Nie, Tian, Taylor & Zou (2018) on the bias of sample means under adaptive
sampling; Hadad, Hirshberg, Zhan, Wager & Athey (2021) on inference with
adaptively collected data; Andrews, Kitagawa & McCloskey (2024), "Inference
on winners".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, stats

METHODS = ("ips", "snips", "dr")
_Z = float(stats.norm.ppf(0.975))


@dataclass(frozen=True)
class Readout:
    """A lift estimate with a normal-approximation standard error."""

    estimate: float
    se: float

    def ci(self, z: float = _Z) -> tuple[float, float]:
        return self.estimate - z * self.se, self.estimate + z * self.se

    def covers(self, truth: float) -> bool:
        return abs(self.estimate - truth) <= _Z * self.se

    def claims_win(self) -> bool:
        """True when the two-sided 95% interval lies entirely above zero."""
        return self.estimate - _Z * self.se > 0


def _check_log(arms: np.ndarray, rewards: np.ndarray,
               propensities: np.ndarray | None = None) -> None:
    if arms.ndim != 1 or arms.shape != rewards.shape:
        raise ValueError("arms and rewards must be 1-D arrays of the same length")
    if len(arms) < 2:
        raise ValueError("need at least 2 impressions")
    if propensities is not None:
        if propensities.shape != arms.shape:
            raise ValueError("propensities must have the same length as arms")
        if np.any(propensities <= 0) or np.any(propensities > 1):
            raise ValueError("propensities of logged arms must be in (0, 1]")


def policy_terms(
    arms: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    target: int,
    method: str = "ips",
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Per-impression terms whose mean estimates the value of always showing ``target``.

    ``propensities`` is the logging probability of the arm actually shown.
    For ``"snips"`` the terms are the linearisation of the ratio estimator, so
    their mean is the SNIPS estimate and their spread gives its delta-method SE.
    ``"dr"`` needs ``baseline``: a prediction of ``target``'s outcome for each
    impression that did not use that impression's outcome.
    """
    arms = np.asarray(arms)
    rewards = np.asarray(rewards, dtype=float)
    propensities = np.asarray(propensities, dtype=float)
    _check_log(arms, rewards, propensities)
    hit = arms == target
    w = np.where(hit, 1.0 / propensities, 0.0)
    if method == "ips":
        return w * rewards
    if method == "snips":
        w_bar = w.mean()
        if w_bar == 0:
            raise ValueError(f"arm {target} never appears in the log")
        v = float((w * rewards).sum() / w.sum())
        return v + w * (rewards - v) / w_bar
    if method == "dr":
        if baseline is None:
            raise ValueError("method='dr' needs a baseline prediction per impression")
        m = np.asarray(baseline, dtype=float)
        if m.shape != arms.shape:
            raise ValueError("baseline must have the same length as arms")
        return m + w * (rewards - m)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def ope_lift(
    arms: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    target: int,
    control: int = 0,
    method: str = "ips",
    baseline_target: np.ndarray | None = None,
    baseline_control: np.ndarray | None = None,
) -> Readout:
    """Off-policy estimate of ``value(target) - value(control)`` from a logged test.

    Valid for a target chosen *before* looking at these rows. If the same rows
    picked the target, the estimate inherits the winner's curse.
    The SE uses the per-impression differences, which is valid when the logged
    propensities are bounded away from zero.
    """
    d = (policy_terms(arms, rewards, propensities, target, method, baseline_target)
         - policy_terms(arms, rewards, propensities, control, method, baseline_control))
    return Readout(float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))


def naive_lift(arms: np.ndarray, rewards: np.ndarray, target: int,
               control: int = 0) -> Readout:
    """Raw mean of ``target`` minus raw mean of ``control``, Welch SE.

    This is what a dashboard reports. It ignores how traffic was allocated.
    """
    arms = np.asarray(arms)
    rewards = np.asarray(rewards, dtype=float)
    _check_log(arms, rewards)
    yt, yc = rewards[arms == target], rewards[arms == control]
    if len(yt) < 2 or len(yc) < 2:
        raise ValueError("each arm needs at least 2 impressions")
    return Readout(float(yt.mean() - yc.mean()),
                   float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))))


def pick_winner(arms: np.ndarray, rewards: np.ndarray, propensities: np.ndarray,
                candidates: np.ndarray | list[int]) -> int:
    """Candidate with the highest SNIPS value. Use it on the selection fold only."""
    best, best_v = -1, -np.inf
    for k in candidates:
        hit = np.asarray(arms) == k
        if not hit.any():
            continue
        w = 1.0 / np.asarray(propensities, dtype=float)[hit]
        v = float((w * np.asarray(rewards, dtype=float)[hit]).sum() / w.sum())
        if v > best_v:
            best, best_v = int(k), v
    if best < 0:
        raise ValueError("no candidate appears in the log")
    return best


# -- minimal data generator ------------------------------------------------------


@dataclass(frozen=True)
class BanditLog:
    """Impression-level log from :func:`simulate_bandit_log`. Arm 0 is the control."""

    arms: np.ndarray
    rewards: np.ndarray
    propensities: np.ndarray
    batch: np.ndarray
    eval_slice: np.ndarray  # True for impressions whose outcomes the bandit never saw
    allocation: np.ndarray  # (n_batches, n_arms) probabilities used for each batch
    posterior_mean: np.ndarray  # (n_batches, n_arms) Beta posterior mean before each batch
    final_posterior_mean: np.ndarray  # (n_arms,) after the last batch
    multiplier: np.ndarray  # (n_batches,) traffic-wide click-rate multiplier

    def baseline(self, arm: int) -> np.ndarray:
        """The bandit's posterior mean for ``arm`` at the time each impression was served."""
        return self.posterior_mean[self.batch, arm]

    def leader(self) -> int:
        """The variant the bandit ends on: highest final posterior mean, control excluded."""
        return int(np.argmax(self.final_posterior_mean[1:]) + 1)


def simulate_bandit_log(
    lifts: np.ndarray | list[float],
    p0: float,
    n_per_batch: int,
    n_batches: int,
    rng: np.random.Generator,
    control_share: float = 0.2,
    floor: float = 0.2,
    drift: float = 0.0,
    eval_share: float = 0.0,
    prior: tuple[float, float] = (1.0, 1.0),
    n_draws: int = 2_000,
) -> BanditLog:
    """Batched Thompson sampling over ``len(lifts)`` variants plus a fixed control share.

    Impression ``i`` in batch ``b`` shown arm ``k`` clicks with probability
    ``p0 * m_b * (1 + lift_k)`` (``lift_0 = 0`` for the control). ``m_b`` rises
    linearly from ``1 - drift`` to ``1 + drift``, so it averages 1 over the test
    and the true relative lift of variant ``k`` is ``lifts[k-1]``. The bandit
    assumes rates are constant, as most production bandits do.

    Within each batch the control gets ``control_share``. The variants share
    the rest: ``(1 - floor)`` by Thompson-sampling probability (estimated from
    ``n_draws`` posterior draws from a Beta(``prior``) start) and ``floor``
    uniformly. The
    logged propensity is the exact probability used to assign the impression.

    A random ``eval_share`` of each batch forms the evaluation slice: those
    impressions are served with the same probabilities, but their outcomes are
    not used to update the bandit.
    """
    lifts = np.asarray(lifts, dtype=float)
    k = len(lifts)
    if k < 1:
        raise ValueError("need at least one variant")
    if not 0.0 < control_share < 1.0:
        raise ValueError("control_share must be in (0, 1)")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0, 1]")
    if not 0.0 <= drift < 1.0:
        raise ValueError("drift must be in [0, 1)")
    if not 0.0 <= eval_share < 1.0:
        raise ValueError("eval_share must be in [0, 1)")
    if min(prior) <= 0:
        raise ValueError("prior Beta parameters must be positive")
    if n_batches < 2 or n_per_batch < 10:
        raise ValueError("need n_batches >= 2 and n_per_batch >= 10")
    mult = 1.0 + drift * np.linspace(-1.0, 1.0, n_batches)
    rel = np.concatenate([[0.0], lifts])
    rates = p0 * mult[:, None] * (1.0 + rel[None, :])
    if np.any(rates < 0) or np.any(rates > 1):
        raise ValueError("p0 * multiplier * (1 + lift) must stay in [0, 1]")

    n_arms = k + 1
    a, b = np.full(n_arms, float(prior[0])), np.full(n_arms, float(prior[1]))
    alloc = np.empty((n_batches, n_arms))
    post = np.empty((n_batches, n_arms))
    arms_out, y_out, ev_out = [], [], []
    for t in range(n_batches):
        post[t] = a / (a + b)
        if t == 0:
            ts = np.full(k, 1.0 / k)
        else:
            draws = rng.beta(a[1:], b[1:], size=(n_draws, k))
            ts = np.bincount(draws.argmax(axis=1), minlength=k) / n_draws
        alloc[t, 0] = control_share
        alloc[t, 1:] = (1.0 - control_share) * ((1.0 - floor) * ts + floor / k)
        arm = rng.choice(n_arms, size=n_per_batch, p=alloc[t])
        y = (rng.random(n_per_batch) < rates[t, arm]).astype(float)
        ev = rng.random(n_per_batch) < eval_share
        learn = ~ev
        clicks = np.bincount(arm[learn], weights=y[learn], minlength=n_arms)
        shows = np.bincount(arm[learn], minlength=n_arms)
        a += clicks
        b += shows - clicks
        arms_out.append(arm)
        y_out.append(y)
        ev_out.append(ev)
    arms = np.concatenate(arms_out)
    batch = np.repeat(np.arange(n_batches), n_per_batch)
    return BanditLog(
        arms=arms,
        rewards=np.concatenate(y_out),
        propensities=alloc[batch, arms],
        batch=batch,
        eval_slice=np.concatenate(ev_out),
        allocation=alloc,
        posterior_mean=post,
        final_posterior_mean=a / (a + b),
        multiplier=mult,
    )


# -- closed-form reference -------------------------------------------------------


def expected_max_std_normal(k: int) -> float:
    """E[max of ``k`` independent standard normals]."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if k == 1:
        return 0.0
    f = lambda z: z * k * stats.norm.pdf(z) * stats.norm.cdf(z) ** (k - 1)  # noqa: E731
    return float(integrate.quad(f, -np.inf, np.inf)[0])


def expected_winner_bias_fixed(k: int, n_per_arm: int, p: float) -> float:
    """Winner's-curse bias of the naive lift when ``k`` equal variants each get a
    fixed ``n_per_arm`` impressions at click rate ``p`` and the highest raw mean
    is reported (all true lifts zero, normal approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return float(np.sqrt(p * (1 - p) / n_per_arm) * expected_max_std_normal(k))
