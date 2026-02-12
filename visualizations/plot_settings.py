import matplotlib
matplotlib.use('pgf')
import numpy as np

def set_latex_settings():
    # Use the pgf backend
    matplotlib.pyplot.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,            # Use LaTeX for text rendering
        "font.family": "serif",         # Use a serif font (matches LaTeX default)
        "font.serif": ["Times New Roman"], # Computer Modern Roman font (Neurips uses Times New Roman)
        "font.size": 10,                # Match LaTeX document font size
        "axes.labelsize": 10,           # Label size
        "axes.titlesize": 10,           # Title size
        "legend.fontsize": 9,           # Legend font size
        "xtick.labelsize": 9,           # X-axis tick size
        "ytick.labelsize": 9,           # Y-axis tick size
        "text.latex.preamble": r"\usepackage{amsmath, amssymb}",  # Custom LaTeX packages
        "pgf.rcfonts": False,           # Don't override rc settings with default pgf settings
    })

# AAI template
column_width = 3.3
document_width = 3.3*2+.375 # excludes left and right margins

def hash(text:str):
  hash=0
  for ch in text:
    hash = ( hash*281 ^ ord(ch)*997) & 0xFFFFFFFF 
  return hash / 2**32 # returns a number between 0 and 1

def format(setting_name:str):
    rng = np.random.default_rng(hash(setting_name)+1)
    color =         {'Unguided Generation': 'k',
                    'Post-Generation TS': 'b',
                    'ToSFiT': 'r',
                    'ToSFiT 1E-6': 'tab:orange',
                    'ToSFiT 1E-7': 'r',
                    'ToSFiT 1E-8': 'tab:purple',
                    'ToSFiT 1': 'r',
                    'ToSFiT 4': 'tab:green',
                    'ToSFiT 16': 'tab:pink',
                    }
    random_color = rng.choice(matplotlib.pyplot.rcParams['axes.prop_cycle'].by_key()['color'])
    line_style =    {'Unguided Generation': '-',
                    'Post-Generation TS': '--',
                    'ToSFiT': ':',
                    'ToSFiT 1E-6': '-.',
                    'ToSFiT 1E-7': ':',
                    'ToSFiT 1E-8': '--',
                    'ToSFiT 1': ':',
                    'ToSFiT 4': '--',
                    'ToSFiT 16': '-',
                    }
    random_line_style = rng.choice(['-', '--', '-.', ':'])

    return {'color': color.get(setting_name, random_color), 'linestyle': line_style.get(setting_name, random_line_style)}
