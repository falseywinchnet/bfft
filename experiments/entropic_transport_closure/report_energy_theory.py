"""Render the retained theory probes; no experiment is rerun."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder=Path(__file__).parent/"results"/"theory"
    data=json.loads((folder/"entropic_energy_theory.json").read_text())
    fig, axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    rows=[r for r in data["long_horizons"] if r["family"]=="small_gap" and r["amplitude"]==.1]
    rows.sort(key=lambda r:r["horizon"])
    for key,label,color in [("linear_max_error","Linear", "#a45335"),("second_jet_max_error","Second jet", "#166678")]:
        axes[0].loglog([r["horizon"] for r in rows],[r[key] for r in rows],"o-",label=label,color=color)
    axes[0].set(xlabel="Horizon = ceil(3 / spectral gap)",ylabel="Maximum oscillation error",title="Shrinking gap, fixed initial amplitude")
    axes[0].legend(); axes[0].grid(alpha=.2,which="both")
    rows=sorted([r for r in data["long_horizons"] if r["mixing"]==.001],key=lambda r:r["amplitude"])
    x=[r["amplitude"] for r in rows]; err=[r["second_jet_max_error"] for r in rows]
    axes[1].loglog(x,err,"o-",label="Measured second jet",color="#166678")
    axes[1].loglog(x,[err[0]*(a/x[0])**3 for a in x],"--",label="Cubic reference",color="#b38c36")
    axes[1].set(xlabel="Initial amplitude",ylabel="Maximum oscillation error",title="Second-jet accuracy over 2,070 steps")
    axes[1].legend(); axes[1].grid(alpha=.2,which="both")
    fig.suptitle("Controlled three-state theory probe · fixed-point reference supplied",fontsize=13)
    fig.savefig(folder/"energy_theory.png",dpi=180)
    plt.close(fig)


if __name__=="__main__": main()
