# CMRP Research Review Protocol

Status: `LOCKED_BEFORE_SAMPLE_LEVEL_REVIEW`

## Scope

This review is a zero-training synthesis of frozen R1 and completed T0
artifacts. It must not alter R1/T0, select a new loss, tune a model, or authorize
T1/R2. Outputs are written only under `research_review/results/`.

## Questions

1. Does lower frozen-head prediction drift under REL correspond to lower
   per-sample error degradation?
2. Is the result better described by output contraction, mean shift, or a
   stability/correctness mismatch?
3. Which claims are supported, bounded, unresolved, or forbidden?
4. Is the evidence sufficient for a method paper, an analysis/mechanism paper,
   or neither without independent confirmation?

## Analysis Units

- Primary experimental replication unit: seed (`n=3`).
- Missing condition is reported explicitly; it is not treated as an
  independent random replicate.
- Clip-level quantities are paired by sample ID.
- MOSEI clips are dependent within a video. Descriptive uncertainty uses a
  deterministic video-cluster bootstrap and never replaces seed replication.

## Registered Sample-level Analyses

For every seed, variant, and missing condition on test:

- absolute full and missing MAE;
- paired missing-minus-full absolute-error change with video-cluster 95% CI;
- full/missing prediction mean and standard deviation;
- missing/full prediction standard-deviation ratio;
- mean prediction shift, full-to-missing slope and correlation;
- mean absolute frozen-head prediction drift.

For paired REL minus H0 comparisons:

- change in robustness degradation;
- change in prediction drift;
- Pearson association between those two per-sample changes;
- fraction of samples whose prediction drift improves while error degradation
  worsens.

The same comparisons are summarized separately for the three text-missing
conditions and the other three conditions by first averaging conditions within
each sample, then bootstrapping videos.

## Interpretation Boundaries

- Lower prediction drift with worse error supports a stability/correctness
  mismatch; it does not establish causality.
- Reduced prediction variance may indicate contraction but does not prove
  representation collapse.
- Failed orthogonal alignment and linear probing only bound the tested recovery
  mechanisms. They do not prove universal irrecoverability.
- T0 test data are exploratory. Any future method chosen from these findings
  requires separately preregistered independent confirmation.

## Decision Rule

The consistency method line remains stopped. Research Review may recommend:

- `ARCHIVE`: evidence is incoherent or too weak;
- `ANALYSIS_PAPER_CANDIDATE`: a coherent bounded mechanism story exists, but
  publication readiness still requires novelty review and external validation;
- `PAPER_READY`: prohibited at this stage because no independent confirmatory
  dataset has been audited under this protocol.
