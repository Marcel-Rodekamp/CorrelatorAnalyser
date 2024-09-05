# we plot using matplot lib
import matplotlib as mpl

import matplotlib.pyplot as plt

# Define the default color set
COLORS = {
    "red": (213 / 255, 0 / 255, 0 / 255),
    "pink": (216 / 255, 27 / 255, 96 / 255),
    "purple": (156 / 255, 39 / 255, 176 / 255),
    "deep purple": (81 / 255, 45 / 255, 168 / 255),
    "indigo": (40 / 255, 53 / 255, 147 / 255),
    "blue": (13 / 255, 71 / 255, 161 / 255),
    "light blue": (3 / 255, 155 / 255, 229 / 255),
    "cyan": (0 / 255, 188 / 255, 212 / 255),
    "teal": (0 / 255, 137 / 255, 123 / 255),
    "green": (56 / 255, 142 / 255, 60 / 255),
    "light green": (124 / 255, 179 / 255, 66 / 255),
    "lime": (205 / 255, 220 / 255, 57 / 255),
    "yellow": (255 / 255, 215 / 255, 20 / 255),
    "amber": (255 / 255, 143 / 255, 0 / 255),
    "orange": (239 / 255, 108 / 255, 0 / 255),
    "deep orange": (216 / 255, 67 / 255, 21 / 255),
    "brown": (93 / 255, 64 / 255, 55 / 255),
    "grey": (97 / 255, 97 / 255, 97 / 255),
    "blue grey": (84 / 255, 110 / 255, 122 / 255),
    "black": (0 / 255, 0 / 255, 0 / 255),
}

# Define a set of use colours with certain intentions

MAIN_COLORS = {
    "primary": COLORS["indigo"],
    "complementary": COLORS["amber"],
    "neutral": COLORS["blue grey"],
    "highlight": COLORS["light blue"],
}

POSITIV_COLORS = {
    "primary": COLORS["green"],
    "complementary": COLORS["red"],
    "neutral": COLORS["blue grey"],
    "highlight": COLORS["lime"],
}

CREATIVE_COLORS = {
    "primary": COLORS["orange"],
    "complementary": COLORS["blue"],
    "neutral": COLORS["blue grey"],
    "highlight": COLORS["yellow"],
}

# set the default color cycle
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(COLORS.values()))

# Define style parameters
styleParams = {
    "axes.titlesize": 34,
    "axes.labelsize": 34,
    "legend.fontsize": 28,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "figure.figsize": (16, 9),
    "text.usetex": False,
    "font.family": "cmr10, serif",
    "text.latex.preamble": r"\usepackage{lmodern}",
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
}

plt.style.use(styleParams)
