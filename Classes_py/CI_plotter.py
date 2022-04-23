import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import csv

class CI_plt():
    def __init__(self, 
                 X, 
                 low, 
                 mean, 
                 upper, 
                 title="Model Comparison", 
                 xlabel="Model", 
                 ylabel="Test R^2", 
                 C=0.95, 
                 n=10, 
                 type1=".", 
                 type2="-", 
                 type3="",
                 ylim=(0,1),
                 grid=True,
                 loc='upper left',
                 size=(20,9),
                 fontsize=18,
                 save=False,
                 dir="",
                 filename="",
                 min=None,
                 max=None,
                 ):
        
        self.X        = X
        self.low      = low
        self.mean     = mean
        self.upper    = upper
        self.title    = title
        self.xlabel   = xlabel
        self.ylabel   = ylabel
        self.C        = C
        self.n        = n
        self.type1    = type1
        self.type2    = type2
        self.type3    = type3
        self.ylim     = ylim
        self.grid     = grid
        self.loc      = loc
        self.size     = size
        self.fontsize = fontsize
        self.save     = save
        self.dir      = dir
        self.filename = filename
        self.min      = min
        self.max      = max

    def plot(self):
        # plt size
        plt.rcParams["figure.figsize"] = self.size
        # plt fontsize
        plt.rcParams.update({'font.size': self.fontsize})
        # create x axis
        X_len = len(self.X)
        X_sup = [i for i in range(0, X_len)]
        # set title
        plt.title(self.title)
        # plot
        plt.plot(X_sup, self.mean,  f"r{self.type1}", markersize=15)
        plt.plot(X_sup, self.low,   f"b{self.type2}", markersize=20)
        plt.plot(X_sup, self.upper, f"b{self.type2}", markersize=20)
        if self.min is not None and self.max is not None:
            plt.plot(X_sup, self.min,   f"g{self.type2}", markersize=15)
            plt.plot(X_sup, self.max,   f"g{self.type2}", markersize=15)
            for x_pos, ymin, ymax in zip(X_sup, self.min, self.max):
                plt.vlines(x_pos, ymin=ymin, ymax=ymax, colors="g")
        for x_pos, ymin, ymax in zip(X_sup, self.low, self.upper):
            plt.vlines(x_pos, ymin=ymin, ymax=ymax, colors="b")
        plt.hlines(np.min(self.low),    0, len(self.low), linestyle="dashed" , alpha = 0.2)
        plt.hlines(np.max(self.upper),  0, len(self.low), linestyle="dashed" , alpha = 0.2)
        # set ticks
        plt.xticks(X_sup, self.X)
        # set labels
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        # create legend        
        red_patch  = mpatches.Patch(color='red', label='mean value')
        blue_patch = mpatches.Patch(color='blue', label=f'C={self.C}, n={self.n}')
        patches = [red_patch, blue_patch]
        if self.min is not None and self.max is not None:
            green_patch  = mpatches.Patch(color='green', label='min/max')
            patches.append(green_patch)
        plt.legend(handles=patches, loc=self.loc, title="Confidence Interval")
        # set ylim
        if self.ylim=="auto":
            plt.ylim( (np.min(self.low)-0.01, np.max(self.upper)+0.01) )
            if self.min is not None and self.max is not None:
                plt.ylim( (np.min(self.min)-0.01, np.max(self.max)+0.01) )
        else:
            plt.ylim(self.ylim)
        # grid
        if self.grid:
            plt.grid(alpha=0.5)
        # save
        if self.save:
            plt.savefig(f"{self.dir}{self.filename}.png")
        # show
        plt.show()

# X = ["Xpresso","Xpresso+miRNA","word2vec+LSTM","transformers"]
# low      = [0.555,0.503,0.603,0.600]
# mean     = [0.559,0.515,0.606,0.603]
# upper    = [0.563,0.526,0.609,0.606]
# title    = "Model Comparison"
# xlabel   = "Model"
# ylabel   = "Test R^2"
# C        = 0.95
# n        = 10
# type1    = "_"
# type2    = "_"
# type3    = ""
# ylim     = (0.5,0.62)
# grid     = True
# loc      = 'upper left'
# size     = (20,9)
# fontsize = 18
# save     = True
# dir      = "CI_plots/"
# filename = "modelcomparison"

# plotter = CI_plt(X, low, mean, upper, 
#                      title, xlabel, ylabel, C, n, 
#                      type1, type2, type3, ylim, grid, 
#                      loc, size, fontsize, save, dir, filename)
# plotter.plot()

# X        = ["Xpresso", "BioLSTM\nDeepLncLoc", "Our_Transformer", "DivideEtimpera", "Transformer\nDeepLncLoc"]  # model names
# low      = [0.5593,      0.6029,               0.5998,            0.5804,           0.6080]                  # CI lower bound
# mean     = [0.5668,      0.6058,               0.6028,            0.5819,           0.6100]                  # CI mean
# upper    = [0.5743,      0.6087,               0.6058,            0.5834,           0.6120]                  # CI upper bound
# # min      = [0.55,          0.55,                  0.55,               0.55,              0.55]                     # min
# # max      = [0.615,          0.615,                  0.615,               0.615,              0.615]                     # min   
# title    = "Model Comparison Xpresso Dataset"                   # title
# xlabel   = "Model"                                              # xlabel
# ylabel   = "Test R^2"                                           # ylabel
# C        = 0.95                                                 # level of confidence
# n        = 10                                                   # number of istances
# type1    = "_"                                                  # dash symbol
# type2    = "_"
# type3    = ""
# ylim     = "auto"                                               # limit on y axis
# grid     = True                                                 # plot grid if true
# loc      = 'upper right'                                        # legend location
# size     = (20,9)                                               # plot size
# fontsize = 18                                                   # font size
# save     = False                                                 # save if true
# dir      = "CI_plots/"                                          # directory to save
# filename = title                                                # filename

# plotter = CI_plt(X, low, mean, upper,  
#                      title, xlabel, ylabel, C, n, 
#                      type1, type2, type3, ylim, grid, 
#                      loc, size, fontsize, save, dir, filename)
# plotter.plot()