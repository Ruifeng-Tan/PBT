import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.patches import Patch
from matplotlib.transforms import Affine2D
import seaborn as sns

def radar_factory(num_vars, frame='circle'):

    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize=None):
            self.set_thetagrids(np.degrees(theta), labels)
            if fontsize is not None:
                for label in self.get_xticklabels():
                    label.set_fontsize(fontsize)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def example_data():
    data = [
        ["Package structure\n", f"\n\nChemical            \nsystems            ", "\n\n\nOperation\nTemperatures", "\n\n\nWorking\nconditions", "\n\n     Total\n     cells"],
        ('', [
            [6, 48, 12, 644, 1034],
            [2, 5, 5, 189, 403],
            [1, 1, 1, 77, 77],
            [1, 3, 3, 23, 61],
            [1, 2, 1, 4, 13],
            [1, 1, 1, 1, 8],
            [1, 1, 3, 41, 41],
            [1, 1, 1, 81, 169]]),
    ]
    return data

def normalize_data(data, mins, maxs):
    return [(d - min_val) / (max_val - min_val) for d, min_val, max_val in zip(data, mins, maxs)]

if __name__ == '__main__':
    N = 5
    theta = radar_factory(N, frame='polygon')
    data = example_data()
    spoke_labels = data.pop(0)

    # set the min and max value for each axis
    mins = [0, 0, 0, 0, 0]
    maxs = [6, 48, 11, 644, 1034]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    colors = sns.color_palette()[3:]
    title, case_data = data[0]
    normalized_case_data = [normalize_data(d, mins, maxs) for d in case_data]

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    for label in ax.get_yticklabels():
        label.set_visible(False)
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')
    count = 0
    for d, color, original in zip(normalized_case_data, colors, case_data):
        count = count + 1
        ax.plot(theta, d, '-', color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')

        if count > 2 :
            continue

        for i in range(len(d)):
            if i == 0:
                ax.text(theta[i]-0.08, d[i]+0.05, original[i], color='black', fontsize=15, ha='right')
            elif i == 1:
                ax.text(theta[i]-0.08, d[i]+0.01, original[i], color='black', fontsize=15, ha='right')
            elif i == 2:
                ax.text(theta[i]-0.05, d[i]+0.08, original[i], color='black', fontsize=15, ha='right')
            elif i == 3:
                ax.text(theta[i]+0.14, d[i]+0.14, original[i], color='black', fontsize=15, ha='right')
            elif i == 4:
                ax.text(theta[i]+0.1, d[i]+0.1, original[i], color='black', fontsize=15, ha='right')

    custom_labels = [f"{label}" for label in spoke_labels]
    ax.set_varlabels(custom_labels, fontsize=15)

    labels = ('Ours', 'Ref. Batteryml', 'Ref. HUST', 'Ref. SNL', 'Ref. CALCE', 'Ref. OX', 'Ref. Stanford', 'Ref. MATR')
    legend_elements = [Patch(facecolor=color, edgecolor=color, label=label, lw=1) for label, color in zip(labels, colors)]
    legend = ax.legend(handles=legend_elements, loc=(0.75, .8), labelspacing=0.1, fontsize='x-large')
    legend.set_frame_on(False)
    fig.tight_layout()
    plt.savefig('./figures/second_fig.jpg', dpi=600)
    plt.savefig('./figures/second_fig.pdf', dpi=600)

'''
["Package structure", f"Chemical systems", "Operation Temperature", "Working condition", "Total cells"],
'''
# ours = [6, 48, 12, 644, 1034]
# batteryml = [2, 5, 5, 189, 403]
# # ISU_ILCC = [1, 1, 1, 202, 240]
# MATR = [1, 1, 1, 81, 169]
# Tongji = [1, 1, 1, 11, 130]
# # ZN-coin = [1, 1, 1, 1, 79]
# HUST = [1, 1, 1, 77, 77]
# # BIT2 = [1, 1, 1, 71, 71]
# SNL = [1, 3, 3, 23, 61]
# # MICH = [1, 2, 3, 22, 58]
# # RWTH = [1, 1, 1, 1, 48] dui
# Stanford = [1, 1, 3, 41, 41]
# # XJTU = [1, 1, 1, 2, 23] 
# # HNEI = [1, 1, 1, 2, 14] dui
# CALCE = [1, 2, 1, 4, 13] dui
# # UL_PUR = [1, 1, 1, 1, 10] dui
# OX = [1, 1, 1, 1, 8] dui
