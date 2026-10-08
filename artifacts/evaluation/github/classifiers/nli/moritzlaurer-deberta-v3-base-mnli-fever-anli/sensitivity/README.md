# NLI cutoff sensitivity

This directory does **not** contain another classifier result. It replays the NLI scores already saved for this one model and asks how its results would change under different abstention cutoffs. No model inference or network access occurs.

## How to read the figure

- An entailment score estimates how strongly the text supports a hypothesis such as ‘this repository is a detector’. It is not a probability that the final classification is correct.
- Columns are the minimum entailment score required before emitting a label.
- Rows are the minimum lead required between the best and second-best labels.
- Moving right or upward is stricter and usually creates more abstentions.
- The black outline is the reported setting: score at least 0.5 and lead at least 0.1.
- Blue panels prefer darker cells; red error-rate panels prefer lighter cells.
- The figure shows robustness and trade-offs. It must not be used to tune thresholds on the same evaluation fixture and then report the tuned score as an unbiased result.

Exact values for every tested pair are in `data/nli_threshold_sensitivity.csv`.
