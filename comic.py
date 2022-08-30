import cv2
import numpy as np
import matplotlib.pyplot as plt

#python3 comic.py <name_of_file.extension>
#Click on the save option to save the image
def read_file(filename):
    img=cv2.imread(filename)
    #whenever files are read in cv2 they are usually read 
    #into bgr we will convert that into rgb in below line
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    return img

filename="original.jpg"

img=read_file(filename)

org_img=np.copy(img)

def edge_mask(img,line_size,blur_value):
    '''
        input:Input image
        output:Edges of Images 
    '''
    #convert image from rgb to gray
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray_blur=cv2.medianBlur(gray,blur_value)
    #line_size is thickness of the edges
    edges=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,line_size,blur_value)
    return edges

line_size,blur_value=7,7
edges=edge_mask(img,line_size,blur_value)
# plt.imshow(edges,cmap="binary")
# plt.show()

def color_quantization(img,k):
    #Transform the image
    data=np.float32(img).reshape((-1,3))

    #determine the criteria
    criteria=(cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER,20,0.001)

    #K-means clustering

    ret, label , center = cv2.kmeans(data,k,None, criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)

    result=center[label.flatten()]
    result=result.reshape(img.shape)

    return result

img = color_quantization(img,k=9)
# plt.imshow(img)
# plt.show()

#Reduce the noise
#d is diameter of the pixel 
blurred=cv2.bilateralFilter(img,d=3,sigmaColor=200,sigmaSpace=200)

# plt.imshow(blurred)
# plt.show()

def cartoon():
    c=cv2.bitwise_and(blurred,blurred,mask=edges)
    plt.imshow(c)
    plt.title("Cartoonified Image")
    plt.show()

    # plt.imshow(org_img)
    # plt.title("org_img")
    # plt.show()
    return c

output=cartoon()
isWritten = cv2.imwrite(r'./cartoon.jpg', output)

if isWritten:
	print('The image is successfully saved.')

#have to try line thicknesses,k values,diameter d values for perfect cartoon image


