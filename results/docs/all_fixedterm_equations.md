# Full FixedTermOLS Equations

This document lists the final sparse FixedTermOLS equations for all stages and all six targets. These equations are part of the main interpretability result.

## Stage 4-8

### x_mean

$$
\begin{aligned}
\widehat{x_{\mathrm{mean}}} &= 1.03\,m_x + 0.02\,\langle x_j^2\rangle_N + 1.39\times 10^{-3}\,\langle x_j^3\rangle_N\\&\quad - 3.45\times 10^{-3}\,\langle m_xx_j^2\rangle_N
\end{aligned}
$$

- Selected terms: **4**

- In-stage R2: **0.997**; MSE: **0.343**; WAIC: **1578.714**

### x_half

$$
\begin{aligned}
\widehat{x_{\mathrm{half}}} &= - 1.85\times 10^{-3}\,m_z^2 + 5.21\times 10^{-3}\,\langle (x_j-m_x)^2\rangle_N + 0.11\,\langle (y_j-m_y)^2\rangle_N\\&\quad - 0.41\,\langle x_j\rangle_N + 0.01\,\langle x_j^2\rangle_N + 1.10\times 10^{-3}\,m_x^2m_z\\&\quad + 3.00\times 10^{-3}\,\langle (m_yy_j)^3\rangle_N
\end{aligned}
$$

- Selected terms: **7**

- In-stage R2: **0.588**; MSE: **0.574**; WAIC: **2041.462**

### y_mean

$$
\begin{aligned}
\widehat{y_{\mathrm{mean}}} &= - 0.23\,\langle x_j\rangle_N - 0.11\,m_x - 0.09\,m_z\\&\quad - 0.05\,m_ym_z - 1.08\times 10^{-3}\,\langle x_j^2\rangle_N
\end{aligned}
$$

- Selected terms: **5**

- In-stage R2: **0.956**; MSE: **0.099**; WAIC: **476.741**

### y_half

$$
\begin{aligned}
\widehat{y_{\mathrm{half}}} &= 9.79\times 10^{-3}\,\langle (x_j-m_x)^2\rangle_N + 1.57\times 10^{-3}\,\langle m_xx_j^2\rangle_N + 2.15\times 10^{-3}\,m_xm_z^2\\&\quad + 9.44\times 10^{-4}\,\langle x_j^3\rangle_N + 8.69\times 10^{-3}\,\langle x_j^2\rangle_N + 0.05\,\langle m_xx_j\rangle_N
\end{aligned}
$$

- Selected terms: **6**

- In-stage R2: **0.852**; MSE: **0.199**; WAIC: **1096.273**

### z_mean

$$
\begin{aligned}
\widehat{z_{\mathrm{mean}}} &= 1.22\,m_z - 0.14\,\langle (y_j-m_y)^2\rangle_N + 6.56\times 10^{-3}\,m_xm_z^2\\&\quad + 0.01\,\langle m_z^2z_j\rangle_N + 2.95\times 10^{-3}\,(m_y-m_x)^2
\end{aligned}
$$

- Selected terms: **5**

- In-stage R2: **0.992**; MSE: **0.337**; WAIC: **1563.349**

### z_half

$$
\begin{aligned}
\widehat{z_{\mathrm{half}}} &= 9.56\times 10^{-3}\,m_x^2 + 0.02\,\langle m_xx_j\rangle_N + 3.04\times 10^{-3}\,\langle x_j^2\rangle_N\\&\quad - 0.02\,m_xm_z + 1.79\times 10^{-4}\,\langle (m_xx_j)^2\rangle_N + 2.07\times 10^{-5}\,(m_xm_y^2m_z)^2\\&\quad + 3.61\times 10^{-3}\,\langle y_j^3\rangle_N
\end{aligned}
$$

- Selected terms: **7**

- In-stage R2: **0.911**; MSE: **0.242**; WAIC: **1273.697**

## Stage 8-12

### x_mean

$$
\begin{aligned}
\widehat{x_{\mathrm{mean}}} &= 0.63\,m_x - 0.59\,\langle y_j\rangle_N + 1.09\,\langle x_j\rangle_N\\&\quad + 3.19\times 10^{-3}\,\langle m_zz_j^2\rangle_N - 3.96\times 10^{-3}\,m_xm_z
\end{aligned}
$$

- Selected terms: **5**

- In-stage R2: **0.995**; MSE: **0.438**; WAIC: **1765.687**

### x_half

$$
\begin{aligned}
\widehat{x_{\mathrm{half}}} &= 0.12\,\langle (y_j-m_y)^2\rangle_N - 0.06\,\langle z_j\rangle_N - 0.10\,m_y^2\\&\quad - 4.62\times 10^{-3}\,\langle m_zz_j^2\rangle_N + 9.51\times 10^{-3}\,\langle (x_j-m_x)^2\rangle_N - 0.01\,\langle y_j^3\rangle_N
\end{aligned}
$$

- Selected terms: **6**

- In-stage R2: **0.658**; MSE: **0.605**; WAIC: **2048.593**

### y_mean

$$
\begin{aligned}
\widehat{y_{\mathrm{mean}}} &= 1.00\,m_y + 0.01\,\langle m_zz_j^2\rangle_N + 4.37\times 10^{-3}\,\langle m_z^2z_j\rangle_N\\&\quad + 0.24\,\langle m_yy_j\rangle_N - 0.01\,m_ym_z^2 + 7.86\times 10^{-4}\,\langle x_j^3\rangle_N\\&\quad + 2.62\times 10^{-3}\,m_x^2 - 2.07\times 10^{-3}\,m_x^2m_y
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.981**; MSE: **0.306**; WAIC: **1456.952**

### y_half

$$
\begin{aligned}
\widehat{y_{\mathrm{half}}} &= 7.49\times 10^{-3}\,(m_z-m_y)^2 - 0.01\,\langle x_j\rangle_N - 0.01\,m_ym_z\\&\quad + 0.18\,m_z - 1.04\times 10^{-3}\,\langle x_j^3\rangle_N - 1.44\times 10^{-3}\,m_xm_ym_z\\&\quad + 6.95\times 10^{-3}\,\langle m_z^2z_j\rangle_N + 3.51\times 10^{-4}\,\langle (m_zz_j)^2\rangle_N
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.925**; MSE: **0.208**; WAIC: **1119.554**

### z_mean

$$
\begin{aligned}
\widehat{z_{\mathrm{mean}}} &= 0.16\,\langle z_j-m_z\rangle_N + 0.02\,(m_z-m_y)^2 - 0.11\,\langle x_j-m_x\rangle_N\\&\quad - 7.72\times 10^{-4}\,\langle x_j^3\rangle_N - 5.16\times 10^{-3}\,m_z^3 - 2.20\times 10^{-3}\,m_x^2m_z\\&\quad + 1.27\,m_z + 9.02\times 10^{-3}\,\langle x_j^2\rangle_N
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.981**; MSE: **0.474**; WAIC: **1839.137**

### z_half

$$
\begin{aligned}
\widehat{z_{\mathrm{half}}} &= 0.06\,\langle z_j^2\rangle_N + 0.02\,m_xm_z + 7.25\times 10^{-3}\,m_x^2\\&\quad + 0.02\,m_ym_z - 3.09\times 10^{-3}\,(m_y-m_x)^2 + 9.85\times 10^{-3}\,\langle (z_j-m_z)^2\rangle_N\\&\quad + 0.01\,\langle m_z^2z_j\rangle_N + 9.51\times 10^{-4}\,m_x^2m_z
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.833**; MSE: **0.477**; WAIC: **1845.128**

## Stage 12-14

### x_mean

$$
\begin{aligned}
\widehat{x_{\mathrm{mean}}} &= 3.71\times 10^{-4}\,\langle x_j^3\rangle_N + 1.44\,m_x - 8.92\times 10^{-3}\,m_x^3\\&\quad + 0.03\,(m_y-m_x)^2 - 4.83\times 10^{-3}\,m_xm_ym_z - 5.57\times 10^{-3}\,\langle y_j^3\rangle_N\\&\quad - 0.02\,\langle z_j^2\rangle_N + 7.55\times 10^{-3}\,m_xm_y^2
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.992**; MSE: **0.463**; WAIC: **917.293**

### x_half

$$
\begin{aligned}
\widehat{x_{\mathrm{half}}} &= 0.03\,\langle x_j^2\rangle_N + 0.03\,\langle (y_j-m_y)^2\rangle_N - 1.79\times 10^{-7}\,(m_xm_ym_z^2)^2\\&\quad + 2.86\times 10^{-3}\,\langle y_j^3\rangle_N
\end{aligned}
$$

- Selected terms: **4**

- In-stage R2: **0.240**; MSE: **0.358**; WAIC: **797.159**

### y_mean

$$
\begin{aligned}
\widehat{y_{\mathrm{mean}}} &= 0.01\,\langle m_yy_j^2\rangle_N - 0.01\,m_xm_y + 0.05\,\langle z_j^2\rangle_N\\&\quad - 4.16\times 10^{-4}\,\langle x_j^3\rangle_N
\end{aligned}
$$

- Selected terms: **4**

- In-stage R2: **0.897**; MSE: **0.291**; WAIC: **706.906**

### y_half

$$
\begin{aligned}
\widehat{y_{\mathrm{half}}} &= 7.09\times 10^{-3}\,\langle x_j^2\rangle_N + 0.04\,m_x + 6.48\times 10^{-3}\,\langle y_j^3\rangle_N\\&\quad - 2.67\times 10^{-6}\,\langle (m_zz_j)^3\rangle_N - 0.28\,\langle y_j\rangle_N
\end{aligned}
$$

- Selected terms: **5**

- In-stage R2: **0.144**; MSE: **0.179**; WAIC: **496.352**

### z_mean

$$
\begin{aligned}
\widehat{z_{\mathrm{mean}}} &= 0.96\,m_z - 0.04\,\langle m_xx_j\rangle_N + 0.04\,m_z^2\\&\quad + 0.03\,m_xm_z - 3.06\times 10^{-6}\,(m_xm_z)^3 - 0.06\,\langle y_j^2\rangle_N\\&\quad - 0.05\,\langle m_y^2y_j\rangle_N
\end{aligned}
$$

- Selected terms: **7**

- In-stage R2: **0.919**; MSE: **0.410**; WAIC: **862.869**

### z_half

$$
\begin{aligned}
\widehat{z_{\mathrm{half}}} &= 8.64\times 10^{-3}\,\langle x_j^2\rangle_N - 4.69\times 10^{-3}\,m_xm_z + 5.22\times 10^{-3}\,\langle z_j^2\rangle_N\\&\quad + 2.34\times 10^{-3}\,\langle y_j^3\rangle_N + 2.68\times 10^{-3}\,\langle z_j^3\rangle_N - 8.16\times 10^{-4}\,(m_ym_z)^2\\&\quad + 0.03\,m_y^2 - 0.22\,\langle z_j\rangle_N
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.085**; MSE: **0.255**; WAIC: **657.590**

## Stage 14-15

### x_mean

$$
\begin{aligned}
\widehat{x_{\mathrm{mean}}} &= - 0.09\,\langle (x_j-m_x)^2\rangle_N - 1.38\,\langle x_j\rangle_N + 0.05\,\langle y_j^2\rangle_N\\&\quad + 1.24\,m_x + 0.12\,\langle x_j^2\rangle_N
\end{aligned}
$$

- Selected terms: **5**

- In-stage R2: **0.343**; MSE: **0.311**; WAIC: **320.364**

### x_half

$$
\begin{aligned}
\widehat{x_{\mathrm{half}}} &= 0.05\,\langle z_j^2\rangle_N - 0.21\,m_y + 9.16\times 10^{-3}\,\langle x_j^2\rangle_N\\&\quad + 0.55\,\langle z_j\rangle_N + 0.36\,m_z - 6.66\times 10^{-5}\,\langle (m_zz_j)^3\rangle_N\\&\quad + 2.56\times 10^{-3}\,\langle z_j^3\rangle_N
\end{aligned}
$$

- Selected terms: **7**

- In-stage R2: **0.124**; MSE: **0.297**; WAIC: **316.088**

### y_mean

$$
\begin{aligned}
\widehat{y_{\mathrm{mean}}} &= - 0.02\,\langle z_j^2\rangle_N + 0.24\,m_y - 0.07\,\langle y_j^2\rangle_N\\&\quad + 7.85\times 10^{-3}\,\langle y_j^3\rangle_N - 0.45\,\langle x_j\rangle_N - 0.01\,(m_z-m_x)^2
\end{aligned}
$$

- Selected terms: **6**

- In-stage R2: **0.125**; MSE: **0.217**; WAIC: **255.585**

### y_half

$$
\begin{aligned}
\widehat{y_{\mathrm{half}}} &= 0.03\,\langle y_j^2\rangle_N
\end{aligned}
$$

- Selected terms: **1**

- In-stage R2: **0.017**; MSE: **0.158**; WAIC: **186.591**

### z_mean

$$
\begin{aligned}
\widehat{z_{\mathrm{mean}}} &= 0.02\,\langle (z_j-m_z)^2\rangle_N + 9.86\times 10^{-3}\,(m_z-m_y)^2 + 0.02\,\langle y_j^3\rangle_N\\&\quad - 0.08\,\langle m_y^2y_j\rangle_N - 4.84\times 10^{-5}\,\langle (m_zz_j)^3\rangle_N + 0.14\,\langle m_zz_j\rangle_N\\&\quad + 0.29\,m_z
\end{aligned}
$$

- Selected terms: **7**

- In-stage R2: **0.275**; MSE: **0.249**; WAIC: **283.036**

### z_half

$$
\begin{aligned}
\widehat{z_{\mathrm{half}}} &= 0.03\,\langle x_j^2\rangle_N + 6.19\times 10^{-4}\,(m_xm_y)^2 - 3.26\times 10^{-3}\,\langle (z_j-m_z)^3\rangle_N\\&\quad - 1.04\,m_z - 5.45\times 10^{-3}\,(m_xm_z)^2 + 0.05\,m_xm_z^2\\&\quad + 4.48\times 10^{-3}\,m_x^3 - 0.38\,m_x
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.135**; MSE: **0.547**; WAIC: **431.611**

## Stage 15-24

### x_mean

$$
\begin{aligned}
\widehat{x_{\mathrm{mean}}} &= 0.66\,m_x + 4.97\times 10^{-3}\,m_xm_y + 1.43\times 10^{-3}\,\langle (y_j-m_y)^3\rangle_N\\&\quad + 0.06\,\langle y_j^2\rangle_N - 0.17\,\langle z_j\rangle_N + 8.30\times 10^{-3}\,\langle m_zz_j\rangle_N\\&\quad + 0.51\,\langle x_j\rangle_N - 4.05\times 10^{-3}\,(m_y-m_x)^2
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.996**; MSE: **0.558**; WAIC: **3788.680**

### x_half

$$
\begin{aligned}
\widehat{x_{\mathrm{half}}} &= 0.03\,\langle (y_j-m_y)^2\rangle_N + 5.31\times 10^{-3}\,(m_z-m_y)^2 + 0.08\,\langle y_j^2\rangle_N\\&\quad + 4.15\times 10^{-3}\,m_xm_z + 0.21\,\langle y_j\rangle_N - 7.18\times 10^{-3}\,m_x^2\\&\quad + 0.02\,\langle x_j^2\rangle_N + 0.01\,m_xm_y
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.796**; MSE: **0.424**; WAIC: **3330.922**

### y_mean

$$
\begin{aligned}
\widehat{y_{\mathrm{mean}}} &= 0.49\,\langle y_j-m_y\rangle_N + 0.14\,\langle x_j\rangle_N - 0.01\,m_xm_z\\&\quad + 1.66\,m_y - 7.90\times 10^{-3}\,(m_z-m_y)^2 - 3.49\times 10^{-3}\,m_ym_z^2\\&\quad - 1.16\times 10^{-3}\,m_x^2m_y + 0.10\,\langle m_yy_j\rangle_N
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.958**; MSE: **0.715**; WAIC: **4206.016**

### y_half

$$
\begin{aligned}
\widehat{y_{\mathrm{half}}} &= 0.01\,\langle m_xx_j\rangle_N + 0.02\,\langle m_zz_j\rangle_N + 4.06\times 10^{-3}\,m_xm_z\\&\quad + 8.57\times 10^{-4}\,m_x^2m_y - 0.01\,m_y^2 + 1.64\times 10^{-3}\,m_xm_ym_z\\&\quad + 8.54\times 10^{-3}\,m_ym_z
\end{aligned}
$$

- Selected terms: **7**

- In-stage R2: **0.754**; MSE: **0.371**; WAIC: **3103.861**

### z_mean

$$
\begin{aligned}
\widehat{z_{\mathrm{mean}}} &= 0.79\,m_z + 0.02\,m_xm_y + 0.25\,m_y\\&\quad + 0.77\,\langle z_j\rangle_N + 0.03\,m_y^2 - 5.11\times 10^{-3}\,m_y^2m_z\\&\quad - 2.21\times 10^{-3}\,(m_y-m_x)^2 - 4.69\times 10^{-3}\,m_xm_z
\end{aligned}
$$

- Selected terms: **8**

- In-stage R2: **0.985**; MSE: **0.787**; WAIC: **4365.823**

### z_half

$$
\begin{aligned}
\widehat{z_{\mathrm{half}}} &= 1.83\times 10^{-3}\,\langle x_j^2\rangle_N + 2.00\times 10^{-3}\,(m_z-m_x)^2 + 4.96\times 10^{-4}\,m_xm_z^2\\&\quad + 9.94\times 10^{-5}\,m_x^3 + 2.56\times 10^{-3}\,(m_y-m_x)^2 + 0.04\,\langle y_j^2\rangle_N
\end{aligned}
$$

- Selected terms: **6**

- In-stage R2: **0.631**; MSE: **0.353**; WAIC: **3020.761**
