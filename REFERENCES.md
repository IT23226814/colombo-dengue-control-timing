# References

IEEE style. The numbers are the ones used in the notebook (Section 7) and in [THEORY.md](THEORY.md).

Every reference was checked against its source:
- For journal articles and books, the authors, title, journal or publisher, volume, issue, pages, year and DOI were compared with the publisher's metadata (Crossref).
- For edition numbers, the publisher's book listing was also checked.
- For data sources and web pages, the page or file itself was opened.

The *Used for* column says what each reference supports in this project.

## Data sources

| # | Reference | Used for |
|---|---|---|
| 1 | National Dengue Control Unit, Ministry of Health, Sri Lanka, "Weekly Dengue Update," 2026, weeks 1–45. [Online]. Available: https://www.dengue.health.gov.lk/weekly-report/ (accessed Nov. 15, 2026). | Weekly suspected dengue cases in Colombo District (`data/colombo_weekly_cases.csv`); the 45 report URLs are in `data/sources/weekly_report_urls.txt` |
| 2 | Department of Census and Statistics, Sri Lanka, "Census of Population and Housing 2024," final report, 2026. [Online]. Available: https://www.statistics.gov.lk/Blog/Blog_details (accessed Nov. 15, 2026). | Population of Colombo District, $N$ = 2,375,415 |
| 9 | Open-Meteo, "Historical Weather API," ERA5-based reanalysis, daily precipitation, 2025–2026, licence CC BY 4.0. [Online]. Available: https://open-meteo.com/en/docs/historical-weather-api (accessed Nov. 18, 2026). | Weekly rainfall for the host–vector model (`data/colombo_weekly_rainfall.csv`) |

## Dengue biology and epidemiology

| # | Reference | Used for |
|---|---|---|
| 3 | M. Chan and M. A. Johansson, "The incubation periods of dengue viruses," *PLoS ONE*, vol. 7, no. 11, p. e50972, 2012, doi: 10.1371/journal.pone.0050972. | Intrinsic incubation period, 5.9 days |
| 4 | World Health Organization, "Dengue and severe dengue," fact sheet, Aug. 21, 2025. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue (accessed Nov. 16, 2026). | Infectious (viraemic) period of about 5 days; extrinsic incubation of 8–12 days |
| 5 | E. A. Undurraga, Y. A. Halasa and D. S. Shepard, "Use of expansion factors to estimate the burden of dengue in Southeast Asia: a systematic analysis," *PLoS Neglected Tropical Diseases*, vol. 7, no. 2, p. e2056, 2013, doi: 10.1371/journal.pntd.0002056. | Prior on the reporting rate $\rho$ (about 13% of symptomatic episodes reported) |
| 8 | O. J. Brady, M. A. Johansson, C. A. Guerra, S. Bhatt *et al.*, "Modelling adult *Aedes aegypti* and *Aedes albopictus* survival at different temperatures in laboratory and field settings," *Parasites & Vectors*, vol. 6, p. 351, 2013, doi: 10.1186/1756-3305-6-351. | Adult mosquito mortality in the host–vector model (0.064 + 0.179 per day) |

## Epidemic and dengue modelling

| # | Reference | Used for |
|---|---|---|
| 10 | W. O. Kermack and A. G. McKendrick, "A contribution to the mathematical theory of epidemics," *Proceedings of the Royal Society of London. Series A*, vol. 115, no. 772, pp. 700–721, 1927, doi: 10.1098/rspa.1927.0118. | Origin of the compartmental (SIR-type) epidemic model |
| 11 | M. J. Keeling and P. Rohani, *Modeling Infectious Diseases in Humans and Animals*. Princeton, NJ, USA: Princeton University Press, 2008, doi: 10.1515/9781400841035. | SEIR model, reproduction numbers, sinusoidal seasonal forcing |
| 12 | L. Esteva and C. Vargas, "Analysis of a dengue disease transmission model," *Mathematical Biosciences*, vol. 150, no. 2, pp. 131–151, 1998, doi: 10.1016/S0025-5564(98)10003-2. | Host–vector dengue model structure |
| 13 | M. Andraud, N. Hens, C. Marais and P. Beutels, "Dynamic epidemiological models for dengue transmission: a systematic review of structural approaches," *PLoS ONE*, vol. 7, no. 11, p. e49085, 2012, doi: 10.1371/journal.pone.0049085. | Context for host-only versus host–vector model structures |
| 6 | H. J. Wearing, P. Rohani and M. J. Keeling, "Appropriate models for the management of infectious diseases," *PLoS Medicine*, vol. 2, no. 7, p. e174, 2005, doi: 10.1371/journal.pmed.0020174. | Why the assumed latent and infectious periods change inferred reproduction numbers and predicted control effects (sensitivity analysis) |
| 7 | S. Marino, I. B. Hogue, C. J. Ray and D. E. Kirschner, "A methodology for performing global uncertainty and sensitivity analysis in systems biology," *Journal of Theoretical Biology*, vol. 254, no. 1, pp. 178–196, 2008, doi: 10.1016/j.jtbi.2008.04.011. | Partial rank correlation coefficient (PRCC) |

## Numerical methods

| # | Reference | Used for |
|---|---|---|
| 14 | J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*, 3rd ed. Chichester, UK: Wiley, 2016, doi: 10.1002/9781119121534. | Classical fourth-order Runge–Kutta method and its order of accuracy |
| 15 | E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd rev. ed. Berlin, Germany: Springer, 1993, doi: 10.1007/978-3-540-78862-1. | Dormand–Prince 8(5,3) method (DOP853), used as the independent high-accuracy reference in the RK4 convergence test |

## Bayesian inference and MCMC

| # | Reference | Used for |
|---|---|---|
| 16 | N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller and E. Teller, "Equation of state calculations by fast computing machines," *The Journal of Chemical Physics*, vol. 21, no. 6, pp. 1087–1092, 1953, doi: 10.1063/1.1699114. | The Metropolis algorithm |
| 17 | W. K. Hastings, "Monte Carlo sampling methods using Markov chains and their applications," *Biometrika*, vol. 57, no. 1, pp. 97–109, 1970, doi: 10.1093/biomet/57.1.97. | Metropolis–Hastings acceptance rule and detailed balance |
| 18 | G. O. Roberts, A. Gelman and W. R. Gilks, "Weak convergence and optimal scaling of random walk Metropolis algorithms," *The Annals of Applied Probability*, vol. 7, no. 1, pp. 110–120, 1997, doi: 10.1214/aoap/1034625254. | Proposal scale $2.38^2/d$ and target acceptance rate of about 0.25 |
| 19 | H. Haario, E. Saksman and J. Tamminen, "An adaptive Metropolis algorithm," *Bernoulli*, vol. 7, no. 2, pp. 223–242, 2001, doi: 10.2307/3318737. | Adapting the proposal covariance from the chain's own samples (done here during burn-in only) |
| 20 | A. Gelman and D. B. Rubin, "Inference from iterative simulation using multiple sequences," *Statistical Science*, vol. 7, no. 4, pp. 457–472, 1992, doi: 10.1214/ss/1177011136. | Multi-chain convergence diagnostic $\hat R$ |
| 21 | A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari and D. B. Rubin, *Bayesian Data Analysis*, 3rd ed. Boca Raton, FL, USA: Chapman and Hall/CRC, 2013, doi: 10.1201/b16018. | Split-$\hat R$, effective sample size, posterior predictive checks |
| 22 | C. J. Geyer, "Practical Markov chain Monte Carlo," *Statistical Science*, vol. 7, no. 4, pp. 473–483, 1992, doi: 10.1214/ss/1177011137. | Initial positive sequence estimator used in the effective sample size |

## Software

| # | Reference | Used for |
|---|---|---|
| 23 | P. Virtanen *et al.*, "SciPy 1.0: fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, no. 3, pp. 261–272, 2020, doi: 10.1038/s41592-019-0686-2. | `scipy.special.gammaln` in the likelihood; `solve_ivp(method="DOP853")` in the RK4 verification only |
| 24 | C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, no. 7825, pp. 357–362, 2020, doi: 10.1038/s41586-020-2649-2. | Vectorised model and sampler code |

## Notes on checking

- **[4]:** the WHO fact sheet's text on viraemia ("about 4–5 days, … as long as 12 days") and on the 8–12 day extrinsic incubation was read on the live page on Nov. 16, 2026. The page shows the date 21 August 2025. When this list was compiled (Nov. 18, 2026) the WHO site returned a server timeout (HTTP 524), so the page could not be reopened that day.
- **[15]:** Crossref lists this DOI under the 1993 second revised edition without author names. The authors and subtitle were taken from the book's listing.
- **[14]:** Crossref's metadata gives no reliable edition number. The 2016 Wiley book is the third edition according to its listing.
- **[1]:** the weekly report PDFs were downloaded and parsed on Nov. 15, 2026 (see `data/README.md`). The report page and the first PDF still opened on Nov. 18, 2026.
