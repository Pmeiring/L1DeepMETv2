from utils import load
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import mplhep as hep
plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")


args = parser.parse_args()
#a=load(args.ckpts + '/' +args.restore_file+ '.resolutions')
a=load(args.ckpts + '/best.resolutions')
colors = {
#    'pfMET': 'black',
   'puppiMET': 'red',
#    'deepMETResponse': 'blue',
#    'deepMETResolution': 'green',
    'MET':  'magenta',
}
label_arr = {
    'MET':     'Graph MET' ,
#    'pfMET':    'PF MET',
   'puppiMET': 'PUPPI MET',
#    'deepMETResponse': 'DeepMETResponse',
#    'deepMETResolution': 'DeepMETResolution',
}
resolutions_arr = {
    'MET':      [[],[],[]],
#    'pfMET':    [[],[],[]],
   'puppiMET': [[],[],[]],
#    'deepMETResponse': [[],[],[]],
#    'deepMETResolution': [[],[],[]],
}
# Define the quantities to plot and figure titles
quantities = [
    ("u_perp_resolution", "u_perp resolution"),
    ("u_perp_scaled_resolution", "u_perp scaled resolution"),
    ("u_par_resolution", "u_par resolution"),
    ("u_par_scaled_resolution", "u_par scaled resolution"),
    ("R", "Response (R)")
]

# Loop over quantities and plot all keys on the same figure
for i, (quantity, title) in enumerate(quantities, start=1):
    plt.figure(i)
    for key in resolutions_arr:
        # Extract data
        x_vals = np.array(a[key][quantity][1])[:40]
        y_vals = np.array(a[key][quantity][0])[:40]

        # Ensure same length
        min_len = min(len(x_vals), len(y_vals))
        x_vals, y_vals = x_vals[:min_len], y_vals[:min_len]

        # Plot
        plt.plot(x_vals, y_vals, color=colors[key], label=label_arr[key])


    plt.title(title)
    plt.xlabel("Bin center")
    plt.ylabel("Resolution")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.ckpts+'/'+quantity+'.png')

# plt.show()
