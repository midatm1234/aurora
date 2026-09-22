# Evaluation against CAMS operational reference data

The importable executor is `aurora_workflow.evaluation.execute_evaluation`.
It reads actual prediction artifacts and writes `metrics.json`, `metrics.svg`,
`evaluation.md` and `evaluation-receipt.json`. The receipt binds executor source,
effective plan, inputs and output hashes; hash consistency is evidence of
artifact integrity and does not itself prove that an unaudited author executed
the stated computation.

The original `evaluate_finetuned_cams_rollouts.ipynb` at fork commit
`88f652f04aa75b65e410b8d00cf4ea07f2edd945` remains a human entry point. It
contains its own catalogue/matching code, ordinary spatial Pearson correlation,
point error metrics and configurable ensemble reduction. Its valid-time
timeseries target matching cannot identify operational forecasts solely by
cycle, and it historically calls the target "truth". The new executor preserves
cycle identity and reports "reference" explicitly. Area weighting is an
explicit versioned evaluation policy, not a claim of bitwise parity with the
notebook's unweighted spatial correlation.

## Portable numeric contract

`baseline.npz`, `refined.npz` and `reference.npz` are NumPy archives opened with
`allow_pickle=False`. Required arrays:

| Key | Shape/meaning |
|---|---|
| fields | `[case,lead,channel,latitude,longitude]`, decoded physical units |
| case_cycle | unique UTC initialization ISO timestamps |
| lead_hours | unique positive 12-hour leads, through 72 |
| channel | unique target names including pressure level, e.g. `no2_1000`, `tcno2` |
| units | explicit physical unit string per channel |
| lat, lon | monotone 1D latitude, increasing regular longitude without duplicate cyclic endpoint |
| mask | optional broadcastable valid-cell boolean mask |
| split | optional train/val/test label per case; must match across products |
| valid_time | optional `[case,lead]`; must equal initialization plus lead |
| ensemble | optional in refined archive: `[member,case,lead,channel,latitude,longitude]` |

Matching uses the intersection of identical cycle/lead keys. Channels, levels,
grids and physical units must match exactly after documented spelling
normalization. No interpolation or unit conversion is guessed. The coverage
report lists missing and excluded cycles/leads; baseline/refined point metrics
share the intersection of finite values and all product masks. Evaluation
defaults to the test split when labels exist. Validation runs must explicitly
select `evaluation.split=val`; test output is not a checkpoint-selection signal.

## Point metrics and aggregation

MAE, RMSE and signed bias (`prediction - reference`) are computed in physical
units per target channel and forecast lead. The spatial weights are spherical
grid-cell areas: the difference of sine latitude edges times the uniform
longitude width. These include correctly clipped polar cells. Errors pool
weighted grid points over selected cases; ordinary weighted spatial correlation
is computed within each case and then averaged over defined cases. Constant
fields or fewer than two matched cells produce undefined correlation, serialized
as JSON `null`. This is not anomaly correlation: no climatology is fitted.

Error improvement is exactly:

```
100 * (baseline_error - refined_error) / baseline_error
```

Zero baseline error gives `null`, and negative percentages mean degradation.
Signed bias is shown directly, without a misleading percent "improvement" of
a signed quantity. Mixed findings remain in the report; higher correlation
alongside worse MAE/RMSE is explicitly flagged.

Results include overall region, interior, boundary, southern boundary,
initialization-hour and valid-hour UTC categories. Boundary means the outer
two grid cells (one on very small grids); the southern diagnostic selects the
southernmost rows irrespective of latitude ordering, making southern-boundary
tcNO2 overcorrection visible. Column and profile channels remain separate.

Coastal masks require supplied aligned land-sea data and are omitted when
unavailable. Hotspots require explicit per-channel physical thresholds through
`evaluation.hotspot_thresholds`; they are omitted without those thresholds.
The reference top decile is a descriptive extreme subset determined per
channel/lead from the matched reference. It is not a training or tuning target.
Global seam jumps and adjacent/patch-boundary differences are reported per
channel in physical units. They are diagnostics, not proof of Transformer
artifacts. The evaluator cannot attribute degradation to a temporal adapter
without separate matched spatial/temporal experiments.

## Stochastic products

Refined `fields` are the selected point product, even when `ensemble` is
present. The evaluator keeps individual-member errors, ensemble-mean errors
and probabilistic scores separate. At least two stochastic members are required
for ensemble diagnostics; one deterministic point cannot claim uncertainty
validation. Complete finite ensembles share one mask, with baseline errors also
reported on that mask.

Empirical finite-ensemble CRPS uses
`mean(|X-y|) - 0.5*mean(|X-X'|)`. A sorted equivalent avoids an allocation
quadratic in member count. This evaluates the empirical distribution, not the
fair/unbiased population estimator. Spread is the square root of area-weighted
population member variance; skill is ensemble-mean RMSE. Their ratio is undefined
when skill is zero. Central 50%, 80% and 90% linear-quantile interval coverage
and mean widths are reported. Fractional rank histograms share tied rank mass
equally. These diagnostics quantify the supplied ensemble and do not establish
general calibration from a small sample.

Historical training objectives are unchanged. The repository's CRPS training
support is discussed in the science contract; an evaluation CRPS implementation
does not imply historical models were trained with CRPS. Different losses must
be separate versioned experiments.

No confidence intervals are currently emitted. Neighboring forecast cases
share input/valid-time support, so counting grid cells as independent would
overstate precision. A future uncertainty analysis must resample appropriately
sized cycle/time blocks and document assumptions. Lack of intervals is reported,
not replaced by independent-grid-cell errors.

## Claims and validation

CAMS operational forecasts/initial states are model references, not independent
observations. Better agreement does not show that Aurora "outperforms CAMS" or
has independently validated air-quality accuracy. Synthetic smoke receipts are
marked `synthetic_fixture`. Historical slide percentages/sample counts are
never inserted into generated metrics.

`tests/workflow/test_evaluation.py` checks known-value errors, percentage signs,
undefined denominators/correlations, area weighting, empirical CRPS, spread,
coverage, tied ranks, matching, units, identical masks, UTC/boundary diagnostics,
and actual report/plot/receipt output. Run lightweight tests with
`python -m pytest --confcutdir=tests/workflow tests/workflow`; this avoids the
legacy repository-wide conftest importing model dependencies for metadata tests.
