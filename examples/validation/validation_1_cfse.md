# Validation 1: CFSE Textbook Table Reproduction

Crystal Field Stabilization Energy (CFSE) and Jahn-Teller (JT)
activity for d$^0$ – d$^{10}$ in octahedral coordination,
compared against standard textbook values.

**Reference**: Miessler & Tarr, *Inorganic Chemistry* (5th ed.);
Chemistry LibreTexts §8.2.2, §5.08

**Formula**: CFSE = n(t$_{2g}$) $\times$ (−0.4Δ$_\mathrm{oct}$)
+ n(e$_g$) $\times$ (+0.6Δ$_\mathrm{oct}$)

## High-Spin Configuration

| d$^n$ | CFSE (calc) | CFSE (ref) | JT (calc) | JT (ref) | Result |
|-------|-------------|------------|-----------|----------|--------|
| d$^{0}$ | 0.0Δ | 0.0Δ | — | — | PASS |
| d$^{1}$ | -0.4Δ | -0.4Δ | weak | weak | PASS |
| d$^{2}$ | -0.8Δ | -0.8Δ | weak | weak | PASS |
| d$^{3}$ | -1.2Δ | -1.2Δ | — | — | PASS |
| d$^{4}$ | -0.6Δ | -0.6Δ | strong | strong | PASS |
| d$^{5}$ | -0.0Δ | 0.0Δ | — | — | PASS |
| d$^{6}$ | -0.4Δ | -0.4Δ | weak | weak | PASS |
| d$^{7}$ | -0.8Δ | -0.8Δ | weak | weak | PASS |
| d$^{8}$ | -1.2Δ | -1.2Δ | — | — | PASS |
| d$^{9}$ | -0.6Δ | -0.6Δ | strong | strong | PASS |
| d$^{10}$ | -0.0Δ | 0.0Δ | — | — | PASS |

## Low-Spin Configuration

| d$^n$ | CFSE (calc) | CFSE (ref) | JT (calc) | JT (ref) | Result |
|-------|-------------|------------|-----------|----------|--------|
| d$^{0}$ | 0.0Δ | 0.0Δ | — | — | PASS |
| d$^{1}$ | -0.4Δ | -0.4Δ | weak | weak | PASS |
| d$^{2}$ | -0.8Δ | -0.8Δ | weak | weak | PASS |
| d$^{3}$ | -1.2Δ | -1.2Δ | — | — | PASS |
| d$^{4}$ | -1.6Δ | -1.6Δ | weak | weak | PASS |
| d$^{5}$ | -2.0Δ | -2.0Δ | weak | weak | PASS |
| d$^{6}$ | -2.4Δ | -2.4Δ | — | — | PASS |
| d$^{7}$ | -1.8Δ | -1.8Δ | strong | strong | PASS |
| d$^{8}$ | -1.2Δ | -1.2Δ | — | — | PASS |
| d$^{9}$ | -0.6Δ | -0.6Δ | strong | strong | PASS |
| d$^{10}$ | -0.0Δ | 0.0Δ | — | — | PASS |

## Summary

**22/22** tests passed.

All CFSE values and Jahn-Teller predictions match textbook references.
