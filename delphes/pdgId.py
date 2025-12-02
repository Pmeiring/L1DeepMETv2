import awkward as ak
import json
import numpy as np

JSON_LOC = "filelist_delphes.json"
dataset = "delphes"  # change as needed

# Load the file list
with open(JSON_LOC, "r") as fo:
    file_names = json.load(fo)[dataset]

all_pdg_ids = set()

for ifile in file_names:
    print(f"Processing file: {ifile}")
    events = ak.from_parquet(ifile)
    for event in events:
        pdg_ids = ak.to_numpy(event["L1T_PUPPIPart_PID"])
        all_pdg_ids.update(pdg_ids)

all_pdg_ids = np.array(sorted(all_pdg_ids), dtype=np.int32)
print("All unique PDG IDs in dataset:", all_pdg_ids)
print("Number of unique PDG IDs:", len(all_pdg_ids))
