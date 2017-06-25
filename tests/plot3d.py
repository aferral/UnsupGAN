# Si quiero hacerlo con hover ver esto
# https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib
import multiprocessing
import pickle
import pylab as plt
import numpy as np
import Tkinter as Tk
from PIL import ImageTk, Image
import os
from matplotlib.widgets import CheckButtons
from infogan.misc.utilsTest import generate_new_color
import sys
from mpl_toolkits.mplot3d import axes3d
import random

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

class GuiApp(object):
    def __init__(self, q):
        self.root = Tk.Tk()
        self.panel = Tk.Label(self.root)
        self.panel.pack()
        self.root.after(100, self.CheckQueuePoll, q)

    def CheckQueuePoll(self, c_queue):
        try:
            path = c_queue.get(0)
            img2 = ImageTk.PhotoImage(Image.open(path))
            self.panel.configure(image=img2)
            self.panel.image = img2

        except Exception:
            pass
        finally:
            self.root.after(100, self.CheckQueuePoll, c_queue)

def onpick(event, queue):
    ind = event.ind[0]
    x, y, z = event.artist._offsets3d
    toOpen=dictXYZname[(x[ind], y[ind], z[ind])]
    print "Point ",x[ind], y[ind], z[ind]," Should open ",toOpen
    queue.put(os.path.join(imageFolder,toOpen))


def setGui(queue):
    gui = GuiApp(queue)
    gui.root.mainloop()

def drawAllfrom(ax,x,y,labelsToDraw):
    colors = []
    plots=[]
    for label in labelsToDraw:
        color=generate_new_color(colors, pastel_factor=0.3)
        colors.append(color)
        pointLabel = x[np.where(y==label)[0]] #Filter rows with points in the given label
        plots.append(ax.scatter(pointLabel[:,0], pointLabel[:,1], pointLabel[:,2],marker=',', c=color, picker=5))
    ax.legend(tuple(plots),map(str,labelsToDraw), scatterpoints=1, loc='upper right',ncol=3,fontsize=8)
    return plots

if __name__ == '__main__':
    args = sys.argv
    if len(args) == 2:
        picklePath = args[2]
        limitPoints = None
    elif len(args) == 3:
        picklePath = args[2]
        limitPoints = args[3]
    else:
        raise Exception("You must give arguments (python plot3d <picklePath> <Maxpoints>(Optional)  ). The pickle format is a list with [x(3D), y, fileListX, nLabels,whereImages]")
    print "About to use ",picklePath," ",limitPoints
    #Cargar datos desde pickle
    with open(picklePath,'rb') as f:
        res = pickle.load(f)
        [x, y, listF, nLabels,imageFolder] = res

    #TODO crear pickle de datos con createExp (crear carpeta)
    #TODO crear un selector de cuantos puntos mostrar

    labels = [i for i in range(nLabels)]

    #TODO Verificar carga de esta cosa (aguanta 10k puntos??)

    dictXYZ={}
    dictXYZname = {}

    #Parse points labels to x,y,z dicts
    # Create dict from x,y,z to iamgeName to be able to open correct image
    for ind,point in enumerate(x):
        dictXYZ[(point[0],point[1],point[2])] = y[ind]
        dictXYZname[(point[0], point[1], point[2])] = listF[ind]


    #Set up GUI for images
    queue = multiprocessing.Queue()
    queue.cancel_join_thread()  # or else thread that puts data will not term

    #Create 3d axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #initial draw
    labelsToDraw = labels
    plots=drawAllfrom(ax,x,y, labelsToDraw)


    #When we click a point show the image asociated
    fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, queue))

    color_check_ax = fig.add_axes([0.025, 0.5, 0.15, 0.15])
    check = CheckButtons(color_check_ax, map(str,labels), tuple([True for i in labels]))

    def func(label):
        print "About to filter by ", label
        selectedPlot = plots[int(label)]
        selectedPlot.set_visible(not selectedPlot.get_visible())
        plt.draw()
    check.on_clicked(func)

    #Start GUI in another process (to be able to plot two things at once)
    t1 = multiprocessing.Process(target=setGui, args=(queue,))
    t1.start()

    #This will keep the plot open until we close it by the X button
    plt.show()

    t1.terminate()
