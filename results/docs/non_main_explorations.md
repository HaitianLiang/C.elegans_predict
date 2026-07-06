# Non-main Explorations

We tested several geometry-aware corrections after the FixedTermOLS baseline. They were useful for understanding failure modes, but they are **not** used as the final model because they did not consistently improve both numeric metrics and 3D morphology.

Non-main attempts included:

- Joint six-target geometric correction;
- weighted latent geometry correction;
- mixed greedy selection with geometric metrics;
- independent direction modeling;
- residual direction / scale calibration;
- radial geometry fitting.

The recurring failure pattern was that a geometry correction could improve one metric while degrading another, or it could expand the predicted point cloud without preserving the correct local direction. The final repository therefore keeps **FixedTermOLS as the main predictive and interpretive model**, and uses geometry metrics for evaluation rather than as a post-hoc correction.

We also validated two additional recurrent-looking channels, $M_{x,2}\to x_{\rm half}$ and $M_{x,2}\to z_{\rm half}$. They are not used as main biological/geometric claims because their marginal correlations change sign across stages. They are better regarded as multivariate equation motifs rather than stable standalone laws.
