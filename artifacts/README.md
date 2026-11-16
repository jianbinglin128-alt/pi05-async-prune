# Frozen channel calibration

channel_importance.npz contains 18 × 16,384 per-channel scores, activation energies, rankings and source bindings.
Collected from 160 raw dense observations, 40 episodes, 20 tasks across four LIBERO suites; activations recomputed with the final Future checkpoint step 29999.
SHA256: 36ab9cc549f1e6e0274745fabcfe0ec08e12d7eeff6430b2c22b5404c32c6803.
Score = balanced mean(z_c²) × sqrt(sum_d W_down[c,d]^4), z=GELU(gate)×up.
No raw images, states or model weights are included. Export verifies source-manifest and score-definition bindings.
