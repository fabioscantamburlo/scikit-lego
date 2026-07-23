# Feature Selection Benchmark Results

**Target Features Selected ($k$):** 5

## Summary Table

| dataset | selector | rf_accuracy | lr_accuracy | stability_jaccard | fit_time_mean |
| --- | --- | --- | --- | --- | --- |
| linear_relevance | AllFeatures | 0.8700 | 0.7933 | 1.0000 | 0.0000 |
| linear_relevance | SelectKBest_ANOVA | 0.8333 | 0.8133 | 0.6524 | 0.0005 |
| linear_relevance | SelectKBest_MI | 0.8633 | 0.8167 | 0.6048 | 0.0309 |
| linear_relevance | mRMR | 0.8033 | 0.7800 | 0.5298 | 0.0005 |
| linear_relevance | JMISelector | 0.8733 | 0.8233 | 0.6857 | 0.0894 |
| linear_relevance | HSICLassoSelector | 0.8500 | 0.8100 | 0.6524 | 0.9143 |
| linear_relevance | FCBFSelector | 0.8667 | 0.8200 | 1.0000 | 0.0223 |
| linear_relevance | MultiSURFSelector | 0.8500 | 0.8133 | 0.7429 | 0.0143 |
| linear_relevance | LaplacianScoreSelector | 0.8133 | 0.8133 | 0.7667 | 0.0436 |
| nonlinear_friedman | AllFeatures | 0.8267 | 0.8233 | 1.0000 | 0.0000 |
| nonlinear_friedman | SelectKBest_ANOVA | 0.8267 | 0.8133 | 0.7667 | 0.0003 |
| nonlinear_friedman | SelectKBest_MI | 0.8067 | 0.7867 | 0.4881 | 0.0187 |
| nonlinear_friedman | mRMR | 0.8267 | 0.8100 | 0.7714 | 0.0005 |
| nonlinear_friedman | JMISelector | 0.8500 | 0.8233 | 0.6762 | 0.0559 |
| nonlinear_friedman | HSICLassoSelector | 0.8233 | 0.8000 | 0.8000 | 0.2815 |
| nonlinear_friedman | FCBFSelector | 0.8567 | 0.8233 | 0.8667 | 0.0189 |
| nonlinear_friedman | MultiSURFSelector | 0.8300 | 0.8233 | 0.7333 | 0.0103 |
| nonlinear_friedman | LaplacianScoreSelector | 0.5500 | 0.4900 | 0.4167 | 0.0196 |
| epistasis_xor | AllFeatures | 0.6625 | 0.4925 | 1.0000 | 0.0000 |
| epistasis_xor | SelectKBest_ANOVA | 0.5625 | 0.5025 | 0.3552 | 0.0002 |
| epistasis_xor | SelectKBest_MI | 0.4825 | 0.4875 | 0.3631 | 0.0198 |
| epistasis_xor | mRMR | 0.5675 | 0.4850 | 0.3313 | 0.0008 |
| epistasis_xor | JMISelector | 0.5200 | 0.4875 | 0.2123 | 0.0722 |
| epistasis_xor | HSICLassoSelector | 0.5950 | 0.5250 | 1.0000 | 0.7249 |
| epistasis_xor | FCBFSelector | 0.5825 | 0.4850 | 0.4643 | 0.0127 |
| epistasis_xor | MultiSURFSelector | 0.9375 | 0.5125 | 0.5810 | 0.0165 |
| epistasis_xor | LaplacianScoreSelector | 0.5050 | 0.5000 | 0.2619 | 0.0189 |
| high_dimensional | AllFeatures | 0.6400 | 0.8400 | 1.0000 | 0.0000 |
| high_dimensional | SelectKBest_ANOVA | 0.9000 | 0.8700 | 0.8667 | 0.0006 |
| high_dimensional | SelectKBest_MI | 0.7000 | 0.7000 | 0.2262 | 0.2744 |
| high_dimensional | mRMR | 0.7600 | 0.7200 | 0.1056 | 0.0017 |
| high_dimensional | JMISelector | 0.7900 | 0.7400 | 0.1306 | 0.9336 |
| high_dimensional | HSICLassoSelector | 0.8400 | 0.8600 | 0.6524 | 0.2659 |
| high_dimensional | FCBFSelector | 0.8000 | 0.7600 | 0.5060 | 0.2149 |
| high_dimensional | MultiSURFSelector | 0.8500 | 0.8300 | 0.5952 | 0.0140 |
| high_dimensional | LaplacianScoreSelector | 0.7600 | 0.8400 | 0.7429 | 0.0474 |
| breast_cancer | AllFeatures | 0.9543 | 0.9491 | 1.0000 | 0.0000 |
| breast_cancer | SelectKBest_ANOVA | 0.9403 | 0.9262 | 0.8667 | 0.0003 |
| breast_cancer | SelectKBest_MI | 0.9350 | 0.9174 | 1.0000 | 0.0288 |
| breast_cancer | mRMR | 0.9403 | 0.9262 | 0.8667 | 0.0006 |
| breast_cancer | JMISelector | 0.9473 | 0.9333 | 0.6714 | 0.1415 |
| breast_cancer | HSICLassoSelector | 0.9438 | 0.9350 | 0.8000 | 2.8707 |
| breast_cancer | FCBFSelector | 0.9491 | 0.9385 | 0.7429 | 0.0237 |
| breast_cancer | MultiSURFSelector | 0.9420 | 0.9332 | 0.7667 | 0.0400 |
| breast_cancer | LaplacianScoreSelector | 0.9209 | 0.9420 | 1.0000 | 0.0698 |
| digits_3v8 | AllFeatures | 0.9888 | 0.9944 | 1.0000 | 0.0000 |
| digits_3v8 | SelectKBest_ANOVA | 0.9636 | 0.9663 | 1.0000 | 0.0004 |
| digits_3v8 | SelectKBest_MI | 0.9607 | 0.9439 | 0.7333 | 0.0482 |
| digits_3v8 | mRMR | 0.9664 | 0.9636 | 0.8000 | 0.0010 |
| digits_3v8 | JMISelector | 0.9664 | 0.9580 | 0.7667 | 0.2024 |
| digits_3v8 | HSICLassoSelector | 0.9636 | 0.9663 | 1.0000 | 1.8344 |
| digits_3v8 | FCBFSelector | 0.9663 | 0.9635 | 0.8000 | 0.0470 |
| digits_3v8 | MultiSURFSelector | 0.9636 | 0.9663 | 1.0000 | 0.0315 |
| digits_3v8 | LaplacianScoreSelector | 0.9776 | 0.9468 | 0.7667 | 0.0636 |
