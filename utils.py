import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def imshowSave(im, path="foo.eps"):
    plt.imshow(im)
    plt.savefig(path)

def showBadCase(sample, trueLabel, predLable, pathPre="./"):
	pass