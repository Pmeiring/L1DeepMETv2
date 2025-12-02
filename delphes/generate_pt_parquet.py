#!/usr/bin/env python3
import os
import time
import json
import torch
import numpy as np
import awkward as ak
from optparse import OptionParser
from torch_geometric.data import Data

JSON_LOC = 'delphes/filelist_delphes.json'

def process_event(event, dataset, currentfile, event_i, outdir):
    """Convert one event into a .pt file."""
    # Convert awkward arrays to numpy
    pt     = np.array(ak.to_list(event["L1T_PUPPIPart_PT"]), dtype=np.float32)
    eta    = np.array(ak.to_list(event["L1T_PUPPIPart_Eta"]), dtype=np.float32)
    phi    = np.array(ak.to_list(event["L1T_PUPPIPart_Phi"]), dtype=np.float32)
    w      = np.array(ak.to_list(event["L1T_PUPPIPart_PuppiW"]), dtype=np.float32)
    pid    = np.array(ak.to_list(event["L1T_PUPPIPart_PID"]), dtype=np.int32)
    charge = np.array(ak.to_list(event["L1T_PUPPIPart_Charge"]), dtype=np.int32)

    # Remove particles with PuppiW == 0
    mask = (w != 0) & (~np.isin(np.abs(pid), [16, 40, 92, 106]))

    pt, eta, phi, w, pid, charge = pt[mask], eta[mask], phi[mask], w[mask], abs(pid[mask]), charge[mask]

    if len(pt) == 0:
        return None

    # Compute px, py
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)

    # Stack features: pt, px, py, eta, phi, puppiWeight, pdgId, charge
    x = np.stack([pt, px, py, eta, phi, w, pid, charge], axis=-1).astype(np.float32)
    x = np.nan_to_num(x)
    x = np.clip(x, -5000., 5000.)

    # Target MET
    met_val = float(ak.to_numpy(event["FullReco_GenMissingET_MET"]))
    phi_val = float(ak.to_numpy(event["FullReco_GenMissingET_Phi"]))
    met_x = met_val * np.cos(phi_val)
    met_y = met_val * np.sin(phi_val)
    y = np.array([[met_x, met_y]], dtype=np.float32)

    # Create PyG Data object
    edge_index = torch.empty((2, 0), dtype=torch.long)
    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.from_numpy(y)
    )

    # Save to .pt
    out_file = os.path.join(outdir, f"{dataset}_file{currentfile}_event{event_i}.pt")
    torch.save(data, out_file)
    return out_file, len(pt)


def main():
    parser = OptionParser()
    parser.add_option('-d', '--dataset', help='dataset', default='delphes', dest='dataset')
    parser.add_option('-s', '--startfile', type=int, default=0, help='startfile index')
    parser.add_option('-e', '--endfile', type=int, default=1, help='endfile index')
    (options, args) = parser.parse_args()

    dataset = options.dataset

    # Read file list
    with open(JSON_LOC, "r") as fo:
        file_names = json.load(fo)[dataset]
    print(f"Found {len(file_names)} files for dataset: {dataset}")

    # Output directory
    outdir = os.path.join(os.environ['PWD'], f"data_{dataset}", "processed_tmp")
    os.makedirs(outdir, exist_ok=True)

    for currentfile, ifile in enumerate(file_names):
        if currentfile < options.startfile:
            continue
        if currentfile >= options.endfile:
            break

        print(f"\nProcessing file {currentfile}: {ifile}")
        events = ak.from_parquet(ifile)
        nevents_total = len(events)
        print(f"Number of events: {nevents_total}")

        for event_i, event in enumerate(events):
            tic = time.time()
            result = process_event(event, dataset, currentfile, event_i, outdir)
            toc = time.time()
            if result is not None:
                out_file, nparticles = result
                print(f"  Saved event {event_i}: {out_file} (nparticles={nparticles}, time={toc-tic:.2f}s)")

    print("\n✅ All done!")


if __name__ == '__main__':
    main()
