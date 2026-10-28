"""Real-multiply counts for the tau-domain FLS inverse, d = 4..15.

Inputs per dimension
  d : dimension
  n : FLS order on B = A tau(A), n = 2^(d//2 - 1)
  c : center size, 1 (scalar) for even d, 2 (scalar + pseudoscalar) for odd d
  s : slot count, length of S0[]; sign table S[][] is s x p
  p : pair count, row length of the pair table T[][] of shape (s-1) x p

Counts are reported in units of one arbitrary multivector product, 4^d real
multiplies.  The model column counts the model's full products: step down,
n sweep products, step up, and E'D for odd d.
"""
import numpy as np

D_MIN, D_MAX = 4, 15
RULES_AFTER = (7, 11)


def tau_domain(d):
    idx = np.arange(1 << d)
    g = np.array([bin(a).count('1') for a in idx])
    rev = 1 - 2 * ((g * (g - 1) // 2) & 1)
    t = max(0, (d - 5) // 2)
    mask = ((1 << t) - 1) << (d - t) if t else 0
    tm = np.array([bin(a & mask).count('1') & 1 for a in idx])
    member = rev * (1 - 2 * tm) == 1
    return member, np.flatnonzero(member)


def inputs(d):
    member, slots = tau_domain(d)
    s = len(slots)
    c = 1 if d % 2 == 0 else 2
    pseudo = (1 << d) - 1
    # unordered pairs {a, b} with a ^ b = k, over non-central output slots k
    rows = {int(member[slots ^ k].sum()) // 2 for k in slots[1:] if k != pseudo}
    assert len(rows) == 1, (d, rows)          # rectangular T[][]
    return d, 1 << (d // 2 - 1), c, s, rows.pop()


def step_down(d, n, c, s, p):
    return (1 << d) + (s - 1) * (1 << (d - 1))


def sweep(d, n, c, s, p):
    return (n - 2) * (c + c * s + 2 * p * (s - c)) + c * s + 2 * (c - 1)


def step_up(d, n, c, s, p):
    return s << d


def model(d, n, c, s, p):
    return n + c + 1          # full products; odd d adds E'D (E'E not counted)


FORMULAS = r"""\begin{align}
M_{\downarrow} &= 2^d + (s-1)\,2^{d-1} \label{eq:step-down}\\
M_{\tau} &= (n-1)\bigl(c + cs + 2p(s-c)\bigr) + cs + 2(c-1) \label{eq:tau-sweep}\\
M_{\uparrow} &= s \cdot 2^d \label{eq:step-up}
\end{align}"""

LEGEND = r"""\begin{minipage}[t]{0.33\linewidth}
\small
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{}ll@{}}
$d$ &= dimensions, \\
$n$ &= FLS order, \\
$c$ &= center size, \\
$s$ &= slots, \\
$p$ &= pairs.
\end{tabular}
\end{minipage}"""


def tabular(spec, header, rows):
    out = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule"]
    for d, line in rows:
        if d - 1 in RULES_AFTER:
            out.append(r"\midrule")
        out.append(line + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    data = [inputs(d) for d in range(D_MIN, D_MAX + 1)]
    f = lambda x: f"{x:,}"
    t_in = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Inputs to the real-product counts formulas by dimension \(d\)}",
        r"\label{tab:tau-inputs}", "",
        r"\begin{minipage}[t]{0.64\linewidth}", r"\centering",
        tabular("rrrrr", r"$d$ & $n$ & $c$ & $s$ & $p$",
                [(r[0], " & ".join(f(x) for x in r)) for r in data]),
        r"\end{minipage}\hfill", LEGEND, "", r"\end{table}"])
    cnt = []
    for r in data:
        u = 4 ** r[0]
        a, b, w = step_down(*r) / u, sweep(*r) / u, step_up(*r) / u
        cells = [f"{x:.2f}" for x in (a, b, w, a + b + w)] + [str(model(*r))]
        cnt.append((r[0], " & ".join([str(r[0])] + cells)))
    t_ct = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Real multiplies per inverse from "
        r"\eqref{eq:step-down}--\eqref{eq:step-up}, in units of one arbitrary "
        r"multivector product ($4^d$ real multiplies), against the model's "
        r"$n+c+1$ products.}",
        r"\label{tab:tau-mults}",
        tabular("rrrrrr",
                r"$d$ & $M_{\downarrow}$ & $M_{\tau}$ & $M_{\uparrow}$ & total & model",
                cnt),
        r"\end{table}"])
    tex = "\n\n".join([t_in, FORMULAS, t_ct]) + "\n"
    open("tau_counts.tex", "w").write(tex)
    print(tex)
