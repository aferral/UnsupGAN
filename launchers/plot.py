import numpy as np

from matplotlib import cm
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh import plotting as bplot
from bokeh.models import (LassoSelectTool, PanTool,
                          ResizeTool, ResetTool,
                          HoverTool, WheelZoomTool,BoxZoomTool)
TOOLS = [LassoSelectTool, PanTool, WheelZoomTool, ResizeTool, ResetTool, BoxZoomTool]
from io import BytesIO
from PIL import Image
import pandas as pd
from skimage import img_as_ubyte
import base64
import pickle
import sys
from infogan.misc.utilsTest import generate_new_color

def to_png(arr):
    out = BytesIO()
    im = Image.fromarray(arr)
    im.save(out, format='png')
    return out.getvalue()
def b64_image_files(images, colormap='magma'):
    cmap = cm.get_cmap(colormap)
    urls = []
    for im in images:
        png = to_png(img_as_ubyte(cmap(im)))
        url = 'data:image/png;base64,' + base64.b64encode(png).decode('utf-8')
        urls.append(url)
    return urls

def main(pklPath,real=False):

    lista = []
    with open(pklPath,'rb') as f:
        lista = pickle.load(f)
    assert(len(lista) == 4)
    savepoints = lista[0]
    savelabels = lista[1]
    iamgeSave = lista[2]
    saverls = lista[3]

    assert (len(savepoints.shape) == 2)
    assert(savepoints.shape[1] == 2)

    tooltip = """
        <div>
            <div>
                <img
                src="@image_files" height="60" alt="image"
                style="float: left; margin: 0px 15px 15px 0px; image-rendering: pixelated;"
                border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px;">@source_filenames</span>
            </div>
        </div>
              """

    # savepoints = np.random.rand(1000,2)
    # savelabels = np.random.randint(0,3,size=(1000))
    # iamgeSave = np.random.rand(1000,96,96)
    # iamgeSave = ['test' for i in range(1000)]
    # saverls = np.random.randint(0,3,size=(1000))


    #POINTS,LABELS,IMAGES
    nPoints = savepoints.shape[0]

    if nPoints > 1000:
        savepoints = savepoints[0:1000]
        savelabels = savelabels[0:1000]
        iamgeSave = iamgeSave[0:1000]
        saverls = saverls[0:1000]
        nPoints = 1000


    df = pd.DataFrame(index=np.arange(nPoints), columns={'z','w','image_files','source_filenames','label'})

    df['z'] = pd.Series(savepoints[:,0], index=df.index)
    df['w'] = pd.Series(savepoints[:,1], index=df.index)


    images = iamgeSave

    filenames = b64_image_files(images)
    df['image_files'] = filenames
    df['source_filenames'] = 'test'

    if real:
        df['label'] = saverls
    else:
        df['label'] = savelabels

    bplot.output_file('plot.html')
    hover0 = HoverTool(tooltips=tooltip)

    tools0 = [t() for t in TOOLS] + [hover0]

    p = figure(plot_width=800, plot_height=800, tools= tools0,
               title="Mouse over the dots")

    colorList = []
    for i in range(0, len(set(df['label']))):
        nc = generate_new_color(colorList, pastel_factor=0.9)
        nc = [nc[0]*255,nc[1]*255,nc[2]*255]
        colorList.append(nc)
    colorList =  ['#%02x%02x%02x' % tuple(x) for x in colorList]
    colorList = np.array(colorList)


    colors = colorList[df['label'].values]
    p.scatter(source=df, x='z', y='w', fill_color=colors,size=10 )
    show(p)

if __name__ == '__main__':
    path = sys.argv[1]
    real = sys.argv[2]
    if real == '1':
        real = True
    else:
        real = False
    main(path, real=real)