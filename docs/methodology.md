# Methodology and Statistical Analysis Notes

## 1. Study design

This project is a secondary analysis of the Indonesia 2023 World Bank Enterprise Survey (WBES). The unit of analysis is the establishment represented in the survey microdata. The analysis is observational and cross-sectional for the relevant survey wave.

The WBES sampling frame and eligibility criteria define the population to which estimates may be generalized. The study should not be described as representative of all Indonesian MSMEs, particularly informal businesses and firms outside the eligible survey frame.

## 2. Analytical objectives

The codebase has developed through several related analytical stages:

- descriptive measurement of perceived access-to-finance obstacles;
- comparisons by existing credit status;
- sensitivity checks across alternative eligibility-based weights;
- analysis of firm, industry, and geographic heterogeneity;
- exploratory multivariable logistic regression;
- missingness, common-sample, and model-sample diagnostics;
- sparse-event and separation diagnostics;
- survey-design and variance-estimation audits; and
- robustness analyses excluding sparse geographic categories or pooling regions.

These stages are related but should not be treated as one fixed specification. Each reported result must identify the exact script, sample, outcome, predictors, weight, and model specification that produced it.

## 3. Outcome construction

A recurring outcome is `major_obstacle`, constructed from the perceived access-to-finance obstacle response `k30`:

- `k30` = 3 or 4: major/very severe obstacle (`major_obstacle = 1`);
- `k30` = 0, 1, or 2: no, minor, or moderate obstacle (`major_obstacle = 0`);
- special, unknown, or inapplicable responses: excluded or treated according to an explicit variable-specific rule.

The binary outcome is a subjective perception measure. It does not measure the amount of unmet credit demand, actual loan rejection, or the causal impact of credit.

Some later specifications use alternative outcome definitions (for example, any obstacle or severe/very severe obstacle). These must be separately named and documented; do not conflate them with `major_obstacle`.

## 4. Credit status and application variables

Earlier project specifications use `k82` for existing credit status, `k162` for application status, `k20a1` for the latest application outcome, and `k17` for reasons for not applying. These names and their coding must be checked against the exact Indonesia 2023 data dictionary and questionnaire before any final interpretation.

In particular, non-application is not synonymous with discouraged borrowing. Reasons such as no need for credit, sufficient internal funds, unfavorable terms, collateral requirements, complex procedures, and expected rejection represent substantively different situations. A discouraged-borrower construct must be operationalized from documented response categories and the applicable questionnaire universe, not inferred from non-application alone.

## 5. Survey weights

The local analysis has used `wstrict`, `wmedian`, and `wweak`. The World Bank Indonesia 2023 catalog describes weights named `wweak`, `wmedian`, and `wstrong`. The local variable `wstrict` must not be silently equated with `wstrong`.

Before final estimation:

1. inspect the exact variable names and labels in the downloaded microdata;
2. retrieve the Indonesia 2023 implementation report and sampling note;
3. reconcile the local script's variable names with the official metadata;
4. identify the target population and eligibility assumption for each weight; and
5. record any renaming or crosswalk in the data documentation.

The World Bank Enterprise Surveys methodology explains that the strict, median, and weak eligibility assumptions can produce different weights. It also notes that published Enterprise Surveys indicators generally use the median-eligibility weight. This is background guidance, not a substitute for validating the precise weight fields and intended estimand in this study.

## 6. Weighted descriptive estimates

For eligible sample \(S\), binary outcome \(y_i\), and selected weight \(w_i\):

\[
\widehat{P}_w = \frac{\sum_{i \in S} w_i y_i}{\sum_{i \in S} w_i}.
\]

Report the following together:

- unweighted number of observations;
- unweighted number of positive outcome events;
- weighted estimate;
- exact weight variable and eligibility interpretation;
- sample inclusion/exclusion criteria; and
- whether the statistic is descriptive or inferential.

A weighted point estimate is not a complete survey estimate. Standard errors, confidence intervals, and tests require a validated design-based variance procedure.

## 7. Analytical samples and missingness

The project contains full-sample, common-sample, and model-specific sample audits. These samples can differ because of outcome validity, predictor missingness, weight eligibility, or model requirements.

Every table should identify its own denominator. Do not compare estimates as if they were based on the same population when their analytic samples differ. Complete-case analysis may induce selection bias if inclusion is associated with the outcome or predictors. Missingness diagnostics should inform, but do not automatically resolve, this concern.

## 8. Regression and separation

Logistic regression is used as an exploratory tool for binary outcomes. Interpret coefficients as conditional associations on the model's specified sample and covariates.

Sparse events, zero-event categories, quasi-complete separation, collinearity, and unstable coefficients can undermine conventional maximum-likelihood inference. A model flagged as inferentially unsuitable must not be presented with ordinary p-values or confidence intervals as if those estimates were reliable. Consider substantively justified category consolidation, penalized methods, or a descriptive-only treatment, while documenting any changes.

Model fit statistics do not remedy a misspecified survey design or sparse-data bias.

## 9. Survey-design inference

Before design-based inference, document and validate:

- the official stratum identifier;
- primary sampling unit definition, if applicable;
- sampling weights and their target population;
- any finite population correction;
- panel treatment and the relationship between panel and stratum variables;
- singleton-stratum handling;
- variance estimator (for example, Taylor linearization or replicate weights); and
- degrees of freedom used for tests and confidence intervals.

The local audit history includes many singleton strata and a discrepancy between calculated and model-reported design degrees of freedom. These are unresolved validation issues, not merely cosmetic diagnostics. Do not treat a successful software execution or a matching point estimate as proof that the variance estimator is correct.

A set of computationally validated joint Wald calculations establishes consistency of those calculations under the supplied covariance matrix; it does not independently validate the survey design or covariance estimator.

## 10. Regional and industry comparisons

Regional and industry analyses should:

- use labels verified against official survey metadata;
- report event counts and sample sizes by category;
- identify sparse categories before fitting or testing models;
- avoid overinterpreting categories with very few events;
- document any pooling or exclusions and their substantive rationale; and
- distinguish effect-size descriptions from inferential claims.

Robustness checks that exclude sparse regions or pool regions answer a modified question. They should be presented as sensitivity analyses, not as replacements chosen solely to obtain statistical significance.

## 11. Interpretation rules

Use language such as:

- “is associated with”;
- “the weighted descriptive prevalence is higher/lower”;
- “the pattern is consistent across the tested specifications”; and
- “the estimate is exploratory and subject to survey-design validation.”

Avoid language such as:

- “credit causes financing constraints”;
- “the model proves”;
- “statistically significant” when valid design-based inference has not been established; or
- “representative of all Indonesian MSMEs.”

## 12. Minimum reporting checklist

For each reported model or estimate, disclose:

1. source dataset and version;
2. outcome definition and coding;
3. predictor definitions and coding;
4. analytic sample and exclusions;
5. unweighted N and event count;
6. weight variable and justification;
7. survey-design specification;
8. estimator and variance method;
9. missing-data handling;
10. sparse-event/separation diagnostics;
11. robustness specifications; and
12. limitations on causal interpretation and external validity.
