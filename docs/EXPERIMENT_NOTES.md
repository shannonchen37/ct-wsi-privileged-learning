# Experiment notes

Early experiments tested direct latent alignment, multi-kernel alignment, generic logit distillation, and several relation-matching objectives. These approaches provided limited or inconsistent CT diagnosis gains, and stronger representation alignment could damage diagnosis or anchor semantics.

The final study therefore focuses on the setting where the richer pathology modality has the clearest practical role: completing missing task supervision during training. The final controls separate three effects:

- clinical auxiliary supervision alone;
- generic CT self-training;
- pseudo supervision from a budget-matched pathology teacher.

The improvement is concentrated at the 25% diagnosis-label budget and diminishes at higher budgets. This trend is reported directly rather than hidden by selecting only the favorable setting.

