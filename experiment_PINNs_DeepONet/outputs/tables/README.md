# Table-ready manuscript results

These CSV files record numerical values that are quoted in the manuscript but are not all recoverable from the figure arrays alone.

| File | Contents and provenance |
|---|---|
| `table_4_seed_sensitivity.csv` | Best-checkpoint and final-epoch results for seeds 1234, 3456, 5678, 6789, and 8765. The seed-1234 logs are the original manuscript runs; the other four runs were made on Server 1 in `nvcr.io/nvidia/physicsnemo/physicsnemo:25.06`. |
| `table_4_seed_sensitivity_summary.csv` | Arithmetic mean and sample standard deviation of the five best-checkpoint test MSE values used in Table 4. |
| `table_8_deeponet_checkpoint_reevaluation.csv` | Table 8 DeepONet values reevaluated from the saved `pinns_inf` checkpoints at each domain's largest training p. The full curves stored in `outputs/expr_6_4_2/*/pinns_inf/npy/mse.npz` were regenerated in the same `ivan_plap` evaluation environment. Checkpoint SHA-256 values are included. |
| `table_9_deeponet_direct_finite_p.csv` | Selected unit-disc DeepONet errors against the exact finite-p solution and the limiting solution. No retraining was performed. |
| `reported_solution_norms.csv` | Normalized discrete L2 norms, `sqrt(mean(u**2))`, on the stated test grids for Tables 4, 6, and 8. |
| `appendix_b_h256_checkpoint_results.csv` | Width-256 Appendix B runs recovered from the archived Server 1 logs. |
| `appendix_c_checkpoint_selection.csv` | Best-checkpoint and final-epoch Appendix C results. This exposes the rows for which the old table mixed final-epoch values into a best-checkpoint comparison. |

The Table 9 evaluator exports through the manuscript cutoff $p=500$ are retained under `table_9_deeponet_source/`. Values marked `diverged` in Appendix C are runs for which no finite best checkpoint or final test MSE was recorded.
