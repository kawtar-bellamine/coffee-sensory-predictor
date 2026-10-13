# Model card: coffee cup profile from product data

**Purpose.** Screen green coffees before tasting: estimate the total cup score and six graded attributes (aroma, flavour, aftertaste, acidity, body, balance) from product data. Not a substitute for a sensory panel.

**Inputs.** Origin, process, variety, species, bean colour, altitude (m), moisture, category 1 and 2 defects, quakers. Missing values are imputed with the training median, with a missing-value indicator.

**Not used, on purpose.** Any score measured during the tasting (the total is their sum), the harvest year (future lots fall outside the training range), the producer name (used only to group lots), the grading organisation (studied separately).

**Training data.** 1,338 Coffee Quality Institute lots graded 2010 to 2018, after the data-quality rules in `coffeepred/validate.py`.

**Models.** Ridge regression for the total score and most attributes, random forest for flavour and body (best on grouped cross-validation).

**Performance on producers never seen.** Total score: R² 0.13 (0.04 to 0.22), mean absolute error 1.72 points. Attributes: R² 0.08 to 0.17. On lots graded in later years: about as good as predicting the training average (R² below zero).

**Uncertainty.** Split-conformal 80% intervals cover 78% of held-out lots (73% to 85% per fold), with a width of about 5 points.

**Known limits.** Small share of variance explained; no transfer across years; scores reflect certification graders, not consumers; grading organisation and producer cannot be separated; Robusta lots are graded under a different protocol.

**Reproducibility.** `python experiments/run_all.py` with seed 0 rebuilds every number; `results/run_manifest.json` records the data hash and package versions.
