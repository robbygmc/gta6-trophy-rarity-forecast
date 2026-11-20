# GTA VI Trophy Rarity Forecast

![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/robbygmc/gta6-trophy-rarity-forecast/rank.svg)
![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/robbygmc/gta6-trophy-rarity-forecast/reward.svg)

Counts + dilution forecaster for GTA VI trophy rarity on Exophase: for every
one of the game's 55 PlayStation 5 trophies, five quantiles of its earn rate
at the case's target.

## The model

An earn rate is earners / players tracked, and in launch week both move
fast — so the program forecasts the two counts separately and divides,
instead of extrapolating the rate itself:

- **Players** `P(t)`: `log P` fitted against `log(1 + hours since launch)`
  with a concave (quadratic, curvature capped) curve, extrapolated to the
  target and never below the last read.
- **Earners** `E_i(t)` per trophy: `log E` fitted against `log(1 + hours)`
  (power law) blended with a saturating curve; the fitted slope shrinks
  toward a tier prior (easy bronze fastest, platinum slowest) when the
  trophy's own history is thin; `E` is capped at `P`.
- **Rate** = `E / P`. When the history is flat (launch night: every read
  identical), no trajectory can be fit, so a launch-week **dilution prior**
  takes over: tracked players explode in the first day while earners grow
  only as a fractional power of players, so most rates fall hard. The prior
  is a general two-parameter mechanism calibrated on the public
  launch-night-to-day-2 movement, and a rich history (the final case)
  overrides it entirely through the fits.
- **Spread**: quantiles widen with horizon (random walk on the logit scale)
  plus the fit residual, so the 90% band is honest about how far a rate can
  move. The ~100% trophy stays pinned at 99.99 with a near-zero spread.

## Validation (local)

`../grade.py` (dev-only, not part of the entry) scores the program on the
practice cases (validity, determinism, runtime ~50 ms), on a pseudo-final
holdout, and on two synthetic final-like cases:

| test | this entry | house baseline |
|---|---|---|
| pseudo-final (real data, p01 history → live Exophase rates 25h later) | mean pinball loss **0.352**, median miss 0.070pp, 90% coverage 0.98 | loss **2.261**, miss 0.757pp, coverage 0.07 |
| synthetic rich history (2.5d of reads, target +4.5d, known truth) | loss **0.178**, miss 0.001pp | — |
| synthetic 2 (saturating curves + read noise, off-model) | loss **0.377**, miss 0.509pp, coverage 0.98 | — |

The practice cases themselves have no public answers, so they are used only
for validity and sane spreads — the 90-minute practice loss is, as the rules
warn, little guide to the final's week. The design target is the final's
4.5-day horizon, where the counts + dilution fitter (not the flat-history
prior) does the work.

## Entry

Python 3.13, standard library only, no network, deterministic. Reads every
`*.in` in the input directory, writes one `*.out` per case.

Built by an AI agent for [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)'s
"Forecast how many GTA VI players earn each trophy in week one" bounty.
