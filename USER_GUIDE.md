# User guide: [Project Title] web demo

**Open the demo:** [https://it23226814.github.io/colombo-dengue-control-timing/](https://it23226814.github.io/colombo-dengue-control-timing/), or download the repository and double-click `docs/index.html`. It runs entirely in your browser (current Chrome, Edge or Firefox). Nothing is installed or uploaded.

![Labelled screenshot of the demo](figures/demo-screenshot-labelled.png)

## Controls

| Control | What it does | Range | Default |
|---|---|---|---|
| [Parameter 1] | [meaning, with units] | [min–max] | [value] |
| [Parameter 2] | [...] | [...] | [...] |
| [Parameter 3] | [...] | [...] | [...] |
| Start / Pause | Runs or pauses the simulation | — | — |
| Reset | Clears the run and applies the current settings | — | — |

## Reading the outputs

- **[Visualisation]:** [what the picture shows and how to read it]
- **[Quantitative output]:** [what the number means, its units, and its uncertainty]

## Try this

1. [A short scenario, e.g. "Set the arrival rate to 1.8/min and press Start. The mean wait settles near 4 minutes."]
2. [Change one parameter and say what to look for.]

## Check against the notebook

The box under the results, **Check against the notebook**, always recomputes one reference scenario, whatever the sliders show:
- **Scenario:** the fitted values, with a 40% cut in transmission from week 21 to the end of the year.
- **Result:** the cases prevented, shown beside the value the notebook computes for exactly the same parameters (notebook Section 5.5).

| Reference scenario | Notebook (Section 5.5) | Demo | Tolerance |
|---|---|---|---|
| Fitted values, 40% cut from week 21 to week 52 | 4,665.2 reported cases prevented | 4,665.2 | within 1% (the two agree to better than 0.0001%) |

You can also reproduce it with the sliders. Press *Restore fitted values*, leave the start week at 21, the reduction at 40% and the duration at "rest of the year". The *Cases prevented* box then shows **4,665**.

This is one parameter set. The notebook's posterior median over 2,000 samples for the same scenario is 4,689, with a 95% credible interval of 3,418–6,108.

## Limitations

- [What the demo simplifies compared with the notebook]
- [Parameter ranges where results are not reliable]
