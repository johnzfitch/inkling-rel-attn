"""
A-TIER2 -- analysis of the streaming-pass dumps. Dump-first: reads ONLY
dumps/tier2/*.npz and analysis/round3/head_taxonomy_v2.json. No GPU, no network.

Answers:
  integrity: every dump loads and conserves attention mass.
  (a) seam: bias-attributable step in mass_with(d) across d=1024, per global layer
      and per head, vs Round 4 C1's predicted sign.
  (b) push-outward vs content decay: on RISING heads (Round 3 taxonomy), decompose
      realized attention into content (mass_without) and bias contribution
      (mass_with - mass_without) as a function of d, and report which wins.
"""
import glob
import json
import os

import numpy as np

DUMP = r"R:\inkling\dumps\tier2"
TAX = r"R:\inkling\analysis\round3\head_taxonomy_v2.json"
OUT = r"R:\inkling\analysis\tier2"
GLOBAL = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}
TEXTS = ["01_prose_en", "02_code", "03_templated", "04_multilingual", "05_needles", "06_random"]


def load(layer, text, seq=8192):
    return np.load(os.path.join(DUMP, f"layer{layer:02d}_{text}_s{seq}.npz"), allow_pickle=True)


def integrity():
    files = glob.glob(os.path.join(DUMP, "layer*_s8192.npz"))
    bad = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        m = json.loads(str(d["meta"]))
        n_valid = d["count"].sum()  # total valid (q,k) pairs; sum_d mass == n_queries per head
        tot = d["mass_with"].sum(1)
        # each query's probs sum to 1 over its valid keys -> per head, sum_d mass == n_queries
        exp = tot.mean()
        if np.abs(tot - tot.mean()).max() > 1e-3 or not np.isfinite(d["mass_with"]).all():
            bad.append(os.path.basename(f))
    return len(files), bad


def seam_analysis():
    """(a) + C1 test. Per global layer, bias-attributable step across d=1024."""
    res = {}
    for L in sorted(GLOBAL):
        per_text = {}
        for t in TEXTS:
            d = load(L, t)
            mw = d["mean_mass_with"]        # [H, dmax]
            wo = d["mean_mass_without"]
            mb = d["mean_bias"]
            i, o = slice(1008, 1024), slice(1024, 1040)
            # per-head then average; also keep per-head for the C1 sign test
            with_step = mw[:, i].mean(1) - mw[:, o].mean(1)       # [H]
            without_step = wo[:, i].mean(1) - wo[:, o].mean(1)    # [H]
            bias_attrib = with_step - without_step               # [H]
            bias_in = mb[:, i].mean(1)                            # [H], sign of in-window bias
            per_text[t] = dict(
                bias_attrib_step_mean=float(bias_attrib.mean()),
                without_step_mean=float(without_step.mean()),   # ~0 if content is seam-blind
                heads_positive_frac=float((bias_attrib > 0).mean()),
                bias_in_mean=float(bias_in.mean()),
                bias_out_max=float(np.abs(mb[:, o]).max()),      # must be ~0 (masked)
            )
        res[str(L)] = per_text
    return res


def pushout_analysis():
    """(b) On rising heads: does bias push attention outward, and does it win?"""
    tax = json.load(open(TAX))["trunk_layers"]
    out = {}
    # rising heads live in local layers; use window [1, 512)
    lo, hi = 8, 480     # far-field band, avoid near-field spike and the extent cliff
    for Lk, entry in tax.items():
        L = int(Lk)
        classes = entry["head_class"]                 # list[64] of class strings
        rising = [h for h, c in enumerate(classes) if c == "rising"]
        if not rising:
            continue
        per_text = {}
        for t in TEXTS:
            d = load(L, t)
            mw = d["mean_mass_with"]; wo = d["mean_mass_without"]
            dd = np.arange(mw.shape[1])
            band = (dd >= lo) & (dd < hi)
            x = dd[band].astype(float)
            def slope(y):   # sign of linear trend over the band
                yb = y[:, band]
                return np.polyfit(x, yb.mean(0), 1)[0]
            content_slope = slope(wo)                 # expect < 0 (content decays with d)
            bias_effect = mw - wo                      # bias's contribution to attention
            bias_slope = slope(bias_effect)            # > 0 => bias pushes attention outward
            net_slope = slope(mw)                      # sign of the NET realized attention
            per_text[t] = dict(
                n_rising=len(rising),
                content_slope=float(content_slope),
                bias_effect_slope=float(bias_slope),
                net_slope=float(net_slope),
                bias_pushes_out=bool(bias_slope > 0),
                bias_wins_net=bool(net_slope > 0),
            )
        out[Lk] = per_text
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    n, bad = integrity()
    print(f"integrity: {n} dumps, {len(bad)} bad {'-> '+str(bad) if bad else '(all conserve mass, all finite)'}")

    seam = seam_analysis()
    print("\n(a) SEAM across global layers (bias-attributable step in mass_with across d=1024):")
    print("   layer  bias_attrib_step  without_step  heads>0%  bias_in_sign")
    for L in sorted(GLOBAL):
        vals = seam[str(L)]
        ba = np.mean([v["bias_attrib_step_mean"] for v in vals.values()])
        ws = np.mean([v["without_step_mean"] for v in vals.values()])
        hp = np.mean([v["heads_positive_frac"] for v in vals.values()])
        bi = np.mean([v["bias_in_mean"] for v in vals.values()])
        print(f"   {L:5d}  {ba:+.2e}       {ws:+.2e}    {hp*100:4.0f}%    {bi:+.3f}")

    push = pushout_analysis()
    print("\n(b) PUSH-OUTWARD on rising heads (far-field band d in [8,480)):")
    print("   layer  n_rising  content_slope  bias_effect_slope  net_slope   verdict")
    for Lk in sorted(push, key=int):
        vals = push[Lk]
        cs = np.mean([v["content_slope"] for v in vals.values()])
        bs = np.mean([v["bias_effect_slope"] for v in vals.values()])
        ns = np.mean([v["net_slope"] for v in vals.values()])
        nr = vals[TEXTS[0]]["n_rising"]
        verdict = ("bias pushes out & WINS net" if bs > 0 and ns > 0 else
                   "bias pushes out, content wins net" if bs > 0 else
                   "bias does NOT push out")
        print(f"   {int(Lk):5d}  {nr:6d}   {cs:+.2e}    {bs:+.2e}       {ns:+.2e}  {verdict}")

    json.dump(dict(integrity=dict(n=n, bad=bad), seam=seam, pushout=push),
              open(os.path.join(OUT, "tier2_findings.json"), "w"), indent=2)
    print(f"\nwrote {OUT}\\tier2_findings.json")


if __name__ == "__main__":
    main()
