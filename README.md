# The Biomarker Interaction Trap: A Causal Audit of Spurious Discoveries in NHANES

**Author:** James P. Brody  
*Department of Biomedical Engineering, University of California, Irvine, Irvine, CA 92697, USA*  
**Email:** `jpbrody@uci.edu`  
**Preprint:** [logical_biomarkers_manuscript.pdf](logical_biomarkers_manuscript.pdf)

---

## Overview

Systematic screening of large observational cohorts for non-linear biomarker interactions is a common strategy for discovering disease mechanisms, risk stratification rules, and pharmacological targets. 

Using the National Health and Nutrition Examination Survey (NHANES) as a case study, this repository provides code, data pipelines, and target trial emulations demonstrating that statistically robust, fully replicating biomarker interactions can arise entirely from three pervasive confounding mechanisms:

1. **The Hemoconcentration Trap:** Preanalytical fluid shifts (such as postural transitions or diuretic therapy) that co-regulate circulating macromolecules (e.g., albumin, globulins, cholesterol), creating apparent biological synergies.
2. **The Confounding-by-Indication Trap:** Clinical prescribing guidelines that channel high-risk, multimorbid patients into therapy (e.g., statins, proton pump inhibitors), producing spurious adverse drug-biomarker interactions.
3. **The Demographic Reference Range Trap:** Applying uniform population-wide cutoffs or mathematical ratios to biomarkers that exhibit marked sex dimorphism or age gradients (e.g., uric acid, creatinine, HDL-C, hemoglobin), which sorts participants into demographic subgroups rather than discovering metabolic pathology.

---

## Key Findings

* **Replication is Insufficient:** Each of the three artifacts replicated across independent NHANES survey cycles (discovery: 2015-2018 cycles, validation: 2007-2010 cycles). Replication is a weak test of biological validity when the underlying confounding structure is itself conserved.
* **Causal Auditing Resolves Artifacts:**
  * Adjusting for hydration proxies (serum sodium and hematocrit) eliminates the apparent Albumin x Cholesterol hypertension interaction ($p = 0.412$).
  * The exact same statin-biomarker interaction persists in untreated patients, proving it is a marker of severe diabetic nephropathy rather than a pharmacological side effect.
  * Regressing out age and sex (demographic residualization) attenuates the apparent Uric Acid x Hemoglobin interaction by 74%.

---

## Repository Contents

* `logical_biomarkers_manuscript.tex`: Complete LaTeX source for the publication-grade preprint.
* `logical_biomarkers_manuscript.pdf`: Compiled 12-page preprint document.
* `figure1_three_traps.pdf` / `.png`: Three-panel empirical demonstration of the three traps across discovery and validation cohorts.
* `figure2_demographic_mechanisms.pdf` / `.png`: Multi-panel visualization of demographic sorting and residualization attenuation.
* `validate_comprehensive_gates.py`: Script evaluating multi-marker interaction patterns across independent NHANES cycles.
* `validate_depression_gates.py`: Analysis script for non-linear logic gates.
* `download_merge_validation_comprehensive.py`: NHANES public data extraction and merging pipeline.

---

## Methodological Checklist for Biomarker Interactions

Before treating an automated biomarker interaction as a biological finding, apply this four-step audit:

1. **Demographic Residualization:** Regress out age and sex from all biomarkers before screening ($Z_{i,\text{res}} = B_i - \hat{B}_i(\text{Age}_i, \text{Sex}_i)$).
2. **Hemodynamic Controls:** Include markers of fluid volume and hydration (serum sodium, hematocrit, osmolality) as covariates.
3. **Treatment Stratification:** Test the pattern separately in treated and untreated groups. If the interaction exists in untreated individuals, it is not a drug effect.
4. **Subgroup Size Constraints:** Require adequate sample size in the double-anomaly quadrant ($N_{11} \ge 250$) to prevent small-sample rate instability.

---

## Citation

```bibtex
@article{brody2026biomarker,
  title={The Biomarker Interaction Trap: A Causal Audit of Spurious Discoveries in NHANES},
  author={Brody, James P.},
  journal={Research Square},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```
