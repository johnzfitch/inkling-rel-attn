"""R5-A — the Atlas: one navigable per-layer map of everything measured.

Descriptive glue per the registration: NO new claims; every cell cites the
committed artifact it came from. Activation-derived cells use the certified
A6-corrected widened capture and corrected LF4 instrument. Writes
analysis/round5/atlas.json and the poster figure.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis"
OUT = A / "round5" / "atlas.json"
POSTER = A / "round5" / "atlas_poster.png"
GLOBALS = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    sources = {
        "bandlimit": A / "round4" / "bandlimit.json",
        "nearfield": A / "round4" / "nearfield.json",
        "lf2": A / "round5" / "lf2" / "lf2_knees.json",
        "lf6": A / "round5" / "lf6" / "lf6_mi_mimicry.json",
        "lf7": A / "round5" / "lf7" / "lf7_parentage.json",
        "lf1": A / "round5" / "lf1" / "lf1_pips.json",
        "insitu": A / "round5" / "insitu_corrected" / "insitu_corrected.json",
        "lf4": A / "round5" / "lf4_corrected" / "zoom_lens_corrected.json",
        "depth": A / "round5" / "corpus_v2_corrected" / "depth_readouts.json",
        "figdata": A / "round5" / "corpus_v2_corrected" / "figures" / "figure_data.json",
        "meta": ROOT / "weights" / "_meta.json",
    }
    data = {key: json.loads(path.read_text(encoding="utf-8"))
            for key, path in sources.items()}
    lf2_rows = {f"L{r['layer']:02d}": r for r in data["lf2"]["layers"]}
    fig1 = data["figdata"]["figure_1"]
    depth_effects: dict[str, dict[str, float]] = {}
    for class_name, series in fig1["effects"].items():
        for layer, value in zip(fig1["layers"], series):
            depth_effects.setdefault(f"L{layer:02d}", {})[class_name] = value
    lf4_effects: dict[str, dict[str, float]] = {}
    for row in data["lf4"]["primary"]:
        for layer, value in row["per_layer_effects"].items():
            lf4_effects.setdefault(f"L{int(layer):02d}", {})[row["name"]] = value

    layers = {}
    for layer in range(66):
        name = f"L{layer:02d}"
        meta = data["meta"][str(layer)]
        band = data["bandlimit"]["layers"][name]["0"]
        motif = data["nearfield"]["per_layer_motif_counts"].get(name)
        cell: dict = {
            "scope": "global" if layer in GLOBALS else "local",
            "extent": meta["extent"],
            "f90_cycles_per_token": band["f90_cycles_per_token"],
            "decimate_2x_rel_rmse": band["decimate_2x_rel_rmse"],
            "nearfield_motif_counts": motif,
            "nearfield_selfinclusive_share": (motif[1] / 64 if motif else None),
            "sources": {"f90": "round4/bandlimit.json",
                        "motifs": "round4/nearfield.json"},
        }
        if name in lf2_rows:
            row = lf2_rows[name]
            cell["crest"] = {"breakpoint_d": row["breakpoint"],
                             "significant": row["significant"],
                             "paragraph_scale": row["paragraph_scale"],
                             "source": "round5/lf2/lf2_knees.json"}
        if layer in GLOBALS:
            race = data["lf6"]["family_race"].get(name)
            corr = data["lf6"]["correlations"].get(name)
            if race:
                bics = race["bic"]
                cell["far_field_family"] = {
                    "winner": min(bics, key=lambda m: bics[m]["bic"]
                                  if isinstance(bics[m], dict) else bics[m]),
                    "source": "round5/lf6/lf6_mi_mimicry.json"}
            if corr:
                cell["mi_rank_corr_prose"] = corr["prose"]
            seam_rows = data["insitu"]["seam"][str(layer)]
            in_situ = {
                "seam_bias_attributable_step_mean": float(np.mean([
                    row["bias_attrib_step_mean"] for row in seam_rows.values()
                ])),
                "needle_retrieval": data["insitu"]["needle_summary"][str(layer)],
                "heartbeat_high_head_ratio_q975": data["insitu"][
                    "heartbeat_content_only"
                ]["high_head_ratio_97_5pct_by_layer"][str(layer)],
                "source": "round5/insitu_corrected/insitu_corrected.json",
            }
            if layer == 65:
                in_situ["terminal_wall_with_bias_inside_outside_ratio"] = data[
                    "insitu"
                ]["terminal_wall"]["L65_with_bias_mass_inside_outside_ratio"]
            cell["in_situ_corrected"] = in_situ
        if name in depth_effects:
            cell["aperture_class_effects_corrected"] = {
                **{k: round(v, 4) for k, v in depth_effects[name].items()},
                "source": "round5/corpus_v2_corrected/figures/figure_data.json"}
        if name in lf4_effects:
            cell["lf4_class_effects_corrected"] = {
                **{key: round(value, 6) for key, value in lf4_effects[name].items()},
                "source": "round5/lf4_corrected/zoom_lens_corrected.json",
            }
        layers[name] = cell

    mtp = {m: {"parent_search": "no fork (null-median distances, metrics disagree)",
               "curve_observation": "nearest bank L51/L47, below trunk minimum (unpromoted)",
               "source": "round5/lf7/lf7_parentage.json"}
           for m in [f"M{d}" for d in range(8)]}

    atlas = {"kind": "round5_atlas", "schema_version": 1,
             "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "registration": "ROUND5_LEFTFIELD_SPEC.md R5-A: descriptive glue, no new claims",
             "global_facts": {
                 "pips": "none survive (round5/lf1/)",
                 "oscillation": "none genuine (round4/oscillation_audit.json)",
                 "far_field_geometry": "rise -> paragraph-scale crest -> single-exp tail (lf2+lf6)",
                 "in_situ": "A6-corrected seam/needle/wall/heartbeat readouts (round5/insitu_corrected/)",
                 "null_benchmark": "round5/lf11/bundle.json"},
             "source_sha256": {key: sha256_file(path) for key, path in sources.items()},
             "script_sha256": sha256_file(Path(__file__)),
             "layers": layers, "mtp": mtp}
    OUT.write_text(json.dumps(atlas, indent=2, sort_keys=True), encoding="utf-8")
    print(f"atlas.json: {len(layers)} layers + {len(mtp)} drafters")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    depth_axis = np.arange(66)
    fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
    f90 = [layers[f"L{l:02d}"]["f90_cycles_per_token"] for l in depth_axis]
    axes[0].semilogy(depth_axis, f90, "o-", ms=3, color="#34495e")
    axes[0].set_ylabel("mode-0 f90\n(cycles/token)")
    share = [layers[f"L{l:02d}"]["nearfield_selfinclusive_share"] for l in depth_axis]
    axes[1].plot(depth_axis, share, "o-", ms=3, color="#8e44ad")
    axes[1].set_ylabel("self-inclusive\nmotif share")
    axes[1].axhline(0.5, color="gray", lw=0.6, ls="--")
    crest_x = [l for l in depth_axis if f"L{l:02d}" in lf2_rows]
    crest_y = [lf2_rows[f"L{l:02d}"]["breakpoint"] for l in crest_x]
    crest_sig = [lf2_rows[f"L{l:02d}"]["significant"] for l in crest_x]
    axes[2].scatter(crest_x, crest_y, c=["#c0392b" if s else "#bdc3c7" for s in crest_sig])
    axes[2].axhspan(21, 160, color="orange", alpha=0.12)
    axes[2].set_ylabel("crest position d\n(globals; band =\nparagraph range)")
    rho_x = [l for l in depth_axis if l in GLOBALS]
    rho_y = [layers[f"L{l:02d}"].get("mi_rank_corr_prose") for l in rho_x]
    axes[3].plot(rho_x, rho_y, "s-", color="#16a085")
    axes[3].axhline(0.9, color="gray", lw=0.6, ls="--")
    axes[3].set_ylabel("MI rank-corr\n(prose)")
    ap_x, ap_y = [], []
    for l in depth_axis:
        eff = layers[f"L{l:02d}"].get("aperture_class_effects_corrected", {})
        if "speaker_labels" in eff:
            ap_x.append(l); ap_y.append(eff["speaker_labels"])
    axes[4].bar(ap_x, ap_y, color=["#c0392b" if v > 0 else "#2980b9" for v in ap_y], width=2.2)
    axes[4].axhline(0, color="k", lw=0.6)
    axes[4].set_ylabel("speaker-label\naperture effect\n(corrected)")
    axes[4].set_xlabel("layer")
    for ax in axes:
        for g in GLOBALS:
            ax.axvline(g, color="gold", lw=0.5, alpha=0.5)
    fig.suptitle("Atlas poster v2 — the depth story in five channels "
                 "(gold lines: global layers; every value cites atlas.json)", fontsize=12)
    fig.tight_layout()
    fig.savefig(POSTER, dpi=150)
    print("poster written")


if __name__ == "__main__":
    main()
