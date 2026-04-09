# Pitch-Timbre Coupling Experiment — Task List

## 1. Feature Extraction

For every audio signal (reference violin, DDSP outputs, WORLD outputs):

### 1.1 Extract F0
- Use CREPE, step size matching STFT hop
- Discard frames with confidence < 0.85

### 1.2 Compute F0 derivatives
- Median filter the F0 contour first (window ~5 frames) to suppress estimation jitter
- F0 velocity (central difference): `f0_dot[m] = (f0[m+1] - f0[m-1]) / (2 * dt)`
- F0 acceleration (second order): `f0_ddot[m] = (f0[m+1] - 2*f0[m] + f0[m-1]) / dt^2`
- dt = hop_size / sample_rate

### 1.3 Compute 8 spectral descriptors per frame
From STFT magnitude spectrum `a_k` and normalized spectrum `p_k = a_k / sum(a_k)`:

1. Spectral centroid: `sum(f_k * p_k)`
2. Spectral spread: `sqrt(sum((f_k - centroid)^2 * p_k))`
3. Spectral skewness: `sum((f_k - centroid)^3 * p_k) / spread^3`
4. Spectral decrease: `sum((a_k - a_1)/(k-1) for k>=2) / sum(a_k for k>=2)`
5. Spectral roll-off: frequency where cumulative energy reaches 95%
6. Spectral flatness: `geometric_mean(a_k) / arithmetic_mean(a_k)`
7. Spectral crest: `max(a_k) / mean(a_k)`
8. Spectral flux: `1 - cosine_similarity(a[m-1], a[m])` (undefined at first frame)

Use eps = 1e-10 in flatness to avoid log(0).

### 1.4 Align and store
- Align CREPE timestamps to STFT frames by nearest match
- Each frame becomes one row: f0, f0_dot, f0_ddot, 8 descriptors
- Keep only rows where confidence >= 0.85 and all values finite

---

## 2. Experiment 1: Reference Coupling Model

Input: feature table from reference violin recordings (all frames pooled).

### 2.1 For each descriptor, fit two OLS regressions:

Model A (static): `d_i ~ b0 + b1*f0 + b2*f0^2`

Model B (with dynamics): `d_i ~ b0 + b1*f0 + b2*f0^2 + b3*f0_dot + b4*f0_ddot`

### 2.2 Record per descriptor:
- R² for both models
- Delta R² = R²(B) - R²(A)
- F-test p-value comparing A vs B
- All coefficients and their p-values

### 2.3 Save Model B coefficients — these define the reference coupling function g_i_ref()

---

## 3. Experiment 2: Coupling Reproduction

Input: feature tables from each transfer method + reference coupling coefficients from Experiment 1.

### 3.1 Coupling error
For each method and each descriptor:
- Predict what the reference violin would produce at the output's pitch: `predicted = g_i_ref(f0, f0_dot, f0_ddot)`
- Coupling error: `CE_i = mean((d_i_output - predicted)^2)`

### 3.2 Fit output's own regression
- Same Model B form as Experiment 1
- Record coefficients and R²
- Coefficient distance to reference: `delta_beta_i = norm(beta_output - beta_ref)`

### 3.3 Correlation summary
- Pearson r between each descriptor and f0, per method
- Compare to reference violin's r values

---

## 4. Experiment 3: Dynamics Comparison

Input: Model B coefficients from Experiments 1 and 2.

### 4.1 Build comparison of b3 (velocity) and b4 (acceleration) coefficients across all methods
### 4.2 Flag which coefficients are statistically significant (p < 0.05)

Expected: WORLD b3 and b4 near zero or nonsignificant. DDSP closer to reference.

---

## 5. Plots

### 5.1 Scatter + regression overlay (one per descriptor)
- x: f0, y: descriptor value
- Reference frames as background scatter
- Overlay fitted curves for reference, DDSP, WORLD

### 5.2 Coupling error bar chart
- Grouped by descriptor, one bar per method

### 5.3 Coefficient distance bar chart
- Grouped by descriptor, one bar per method

### 5.4 Correlation comparison
- Pearson r(descriptor, f0) per method, side by side with reference