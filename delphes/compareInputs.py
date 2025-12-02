#!/usr/bin/env python3
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Settings ---
cmssw_folder  = "/mnt/scratch/pmeiring/L1DeepMETv2/L1DeepMETv2/data_ttbar/processed/"
delphes_folder= "/mnt/scratch/pmeiring/L1DeepMETv2/L1DeepMETv2/data_delphes/processed/"
max_events = 1000
n_bins = 50

# ============================================================
#  NPZ PARSER
# ============================================================
def parse_npz_folder(folder, max_events):
    npz_npart, npz_pt, npz_eta, npz_phi = [], [], [], []
    npz_w, npz_pid, npz_charge = [], [], []
    met_x_npz, met_y_npz = [], []

    files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    files.sort()

    for i, file in enumerate(files):
        if i >= max_events:
            break
        print(f"NPZ: {i+1}")

        data = np.load(os.path.join(folder, file))

        x = data["x"]   # [nEvents, nParticles, features]
        y = data["y"]   # [nEvents, 2]

        # Mask out particles with zero weight
        mask = x[:, 3] != 0
        x = x[mask]

        npz_npart.append(len(x))

        npz_pt.extend(x[:,0])
        npz_eta.extend(x[:,1])
        npz_phi.extend(x[:,2])
        npz_w.extend(x[:,3])
        npz_pid.extend(x[:,4])
        npz_charge.extend(x[:,5])

        # MET
        if y.ndim == 2:
            met_x_npz.append(y[0,0])
            met_y_npz.append(y[0,1])
        else:
            met_x_npz.append(y[0])
            met_y_npz.append(y[1])

    return {
        "npart": npz_npart,
        "pt": npz_pt,
        "eta": npz_eta,
        "phi": npz_phi,
        "w": npz_w,
        "pid": npz_pid,
        "charge": npz_charge,
        "met_x": met_x_npz,
        "met_y": met_y_npz
    }


# ============================================================
#  PT PARSER
# ============================================================
def parse_pt_folder(folder, max_events):
    pt_npart, pt_pt, pt_eta, pt_phi = [], [], [], []
    pt_w, pt_pid, pt_charge = [], [], []
    met_x_pt, met_y_pt = [], []

    files = [f for f in os.listdir(folder) 
             if f.endswith(".pt") and not f.startswith("._")]
    files.sort()

    for i, file in enumerate(files):
        if i >= max_events:
            break
        print(f"PT: {i+1}")

        data = torch.load(os.path.join(folder, file))
        x = data["x"]  # [nParticles, features]
        y = data["y"]  # [2] or [1,2]

        if torch.is_tensor(x): x = x.numpy()
        if torch.is_tensor(y): y = y.numpy()

        pt_npart.append(len(x[:,0]))

        pt_pt.extend(x[:,0])
        pt_eta.extend(x[:,3])
        pt_phi.extend(x[:,4])
        pt_w.extend(x[:,5])
        pt_pid.extend(x[:,6])
        pt_charge.extend(x[:,7])

        met_x_pt.extend(y[:,0])
        met_y_pt.extend(y[:,1])

    return {
        "npart": pt_npart,
        "pt": pt_pt,
        "eta": pt_eta,
        "phi": pt_phi,
        "w": pt_w,
        "pid": pt_pid,
        "charge": pt_charge,
        "met_x": met_x_pt,
        "met_y": met_y_pt
    }


# ============================================================
#  HISTOGRAM PLOTTING
# ============================================================
def make_plots(in1, in2, n_bins):
    fig, axes = plt.subplots(3,3, figsize=(18,15))
    axes = axes.flatten()

    def hist(ax, data1, data2, title):
        ax.hist(data1, bins=n_bins, alpha=0.5, color="C0",
                density=True, label="CMSSW Emulator")
        ax.hist(data2, bins=n_bins, alpha=0.5, color="C1",
                density=True, label="Delphes")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend()

    hist(axes[0], in1["npart"], in2["npart"], "Number of L1 PUPPI candidates")
    hist(axes[1], in1["pt"],    in2["pt"],    "L1T_PUPPIPart_PT")
    hist(axes[2], in1["eta"],   in2["eta"],   "L1T_PUPPIPart_Eta")
    hist(axes[3], in1["phi"],   in2["phi"],   "L1T_PUPPIPart_Phi")
    hist(axes[4], in1["w"],     in2["w"],     "L1T_PUPPIPart_PuppiW")
    hist(axes[5], in1["pid"],   in2["pid"],   "L1T_PUPPIPart_PID")
    hist(axes[6], in1["charge"],in2["charge"],"L1T_PUPPIPart_Charge")
    hist(axes[7], in1["met_x"], in2["met_x"], "MET X")
    hist(axes[8], in1["met_y"], in2["met_y"], "MET Y")

    plt.tight_layout()
    plt.savefig("comparison_histograms_lognorm.png")
    print("Saved figure as comparison_histograms_lognorm.png")


# ============================================================
#  MAIN
# ============================================================
def main():
    cmssw_data = parse_pt_folder(cmssw_folder,  max_events)
    delphes_data = parse_pt_folder(delphes_folder,  max_events)
    make_plots(cmssw_data, delphes_data, n_bins)

# Entry point
if __name__ == "__main__":
    main()
