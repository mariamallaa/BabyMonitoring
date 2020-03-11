from commonfunctions import *
from skimage.filters import gaussian
from scipy.ndimage import maximum_filter,gaussian_filter,sobel
import matplotlib.patches as patches
from skimage import io
import numpy as np

def harris(img):
    eps=1e-6
    #np.seterr(divide='ignore', invalid='ignore')

    # TODO perform gaussian_filter:
    image=gaussian(img)
    #show_images([img,image])

    # TODO perform sobel to get x-change and y-change:
    sobelx=sobel_h(image)
    sobely=sobel_v(image)
    #show_images([image,sobelx,sobely])


    # TODO perform gaussian_filter on x-change * x-change to get Wxx:
    x=sobelx*sobelx
    Wxx=gaussian(x)
    # TODO perform gaussian_filter on x-change * y-change to get Wxy:
    xy=sobelx*sobely
    Wxy=gaussian(xy)
    # TODO perform gaussian_filter on y-change * y-change to get Wyy:
    y=sobely*sobely
    Wyy=gaussian(y)

    #print(Wxx,Wyy,Wxy)

    # TODO Get R value (use **2) for power 2:
    R=((Wxx*Wyy)-(Wxy*Wxy))/(Wxx+Wyy+0.000000000001)
    #print(R)
    # TODO Get the R values which are local maximum and above a given threshold
    windowsize=3
    for i in range(0,R.shape[0]-windowsize,windowsize):
        for j in range(0,R.shape[1]-windowsize,windowsize):
            localmax=R[i:i+windowsize,j:j+windowsize]
            arg=np.argmax(localmax)
            #print(arg)
            max=np.max(localmax)
            newwindow=np.zeros([windowsize,windowsize])
            if(max>=0.0006):
                newvector=np.zeros(9)
                newvector[arg]=1
                newwindow=np.reshape(newvector, (3,3))
            R[i:i+windowsize,j:j+windowsize]=newwindow

    #show_images([image,R])
    # This parts will get z. for each z = true, draw a rectangle around the patches
    # Use it as and if needed
    indx = np.argwhere(R==True)

    fig,ax = plt.subplots(1)
    ax.imshow(img)

    for ind in indx:
        rect = patches.Rectangle((ind[1],ind[0]),2,2,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        #print(rect)
    
    plt.show()
