import os
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

# Aaron added
import matplotlib.pyplot as plt
import pdb
from numba import jit
# Only used to provide a standard data structure to speed up clustering
from unionfind import unionfind

if __name__ == '__main__':
    plt.close('all')
    
    ### User-Defined Constants
    ACCEPT = 0.5

def visualize(I, output, disp=False):
    im = Image.fromarray(I)
    draw = ImageDraw.Draw(im)
    for box in output:
        y0, x0, y1, x1, score = box
        draw.rectangle([x0, y0, x1, y1])
        fnt = ImageFont.truetype("arial.ttf", 10)
        draw.text((max(x0-15,0),max(y0-15, 0)), str(score)[:3], font=fnt, fill=(255,255,255))
    if disp:
        im.show()
    return np.asarray(im)

# Aaron: Load the templates to be used by the modified match filter
def loadTemplates():
    kernels = []
    I = np.array(Image.open('data/RedLights2011_Medium/RL-155.jpg'))          
    kernels.append(I[323:345, 206:215, :])
    # Comment out for weak approach
    # kernels.append(I[287:314, 347:356, :])
    # kernels.append(I[332:352, 425:434, :])
    # I = np.array(Image.open('data/RedLights2011_Medium/RL-259.jpg'))          
    # kernels.append(I[220:236, 269:281, :])
    # kernels.append(I[222:238, 315:321, :])
    # I = np.array(Image.open('data/RedLights2011_Medium/RL-062.jpg'))
    # # kernels.append(I[239:254, 157:162, :])
    # kernels.append(I[196:223, 463:475, :])
    # # kernels.append(I[242:251, 349:353, :])
    return kernels

def findEdges(mask, disp=False):
    edgeMask = mask.copy()
    # Classify labeled pixel as edge if it has <= 3 neighbors in 
    # cardinal directions
    numNeighbors = edgeMask[1:-1,2:].astype(int) + edgeMask[1:-1,:-2].astype(int) + edgeMask[:-2,1:-1].astype(int) + edgeMask[2:,1:-1].astype(int)
    newNeighbors = np.zeros(numNeighbors.shape)
    
    # rnd = 0
    while np.sum(np.abs(newNeighbors - numNeighbors)) > 0:
        # print('On iteration', rnd)
        numNeighbors = newNeighbors
        padded = np.zeros(np.shape(edgeMask))
        padded[1:-1,1:-1] = (numNeighbors == 4)
        # Fill in any pixel that is surrounded
        edgeMask += padded.astype(bool) 
        # Repeat now that filled in holes
        newNeighbors = edgeMask[1:-1,2:].astype(int) + edgeMask[1:-1,:-2].astype(int) + edgeMask[:-2,1:-1].astype(int) + edgeMask[2:,1:-1].astype(int)
        # rnd += 1
        
    padded = np.ones(np.shape(edgeMask)) # Enable all pixels on perimeter edge
    padded[1:-1,1:-1] = (newNeighbors <= 3) 
    edgeMask = padded * edgeMask   
    
    if disp:
        plt.figure()
        plt.imshow(edgeMask)
        
    return np.transpose(np.where(edgeMask)), edgeMask    

def findRed(I, disp=False):
    im = Image.fromarray(I)
    hsv = np.asarray(im.convert('HSV'))

    hue = (hsv[:,:,0] < 50)  + (hsv[:,:,0] > 240) > 0
    sat = (hsv[:,:,1] > 130)
    value = (hsv[:,:,2] > 130)
    
    mask = hue * sat * value
    if disp:
        im = Image.fromarray(mask * 255)
        # im.show()
        plt.figure()
        plt.imshow(np.array(im))
    return np.transpose(np.where(mask)), mask

# def heatmapCluster(heatmap, similarityThresh):
#     # Left to right
#     # Up to down
#     # Take the larger of the two as the edge
    
#     original = heatmap[1:-1, 1:-1]
#     right = heatmap[1:-1, 2:]
#     left = heatmap[1:-1, :-2]
#     up = heatmap[:-2, 1:-1]
#     down = heatmap[2:, 1:-1]
    
#     edges = (original - right > similarityThresh) + \
#             (original - left > similarityThresh) + \
#             (original - up > similarityThresh) + \
#             (original - down > similarityThresh)
        
#     pdb.set_trace()
     
# Sliding window clustering where thresh dictates square window size
# mask should give 
def clusterPixels(pixels, mask, thresh):
    
    pixelMap = {tuple(pixels[i]) : i for i in range(len(pixels))} 
    
    u = unionfind(len(pixels))
    for i in range(mask.shape[0] - thresh):
        for j in range(mask.shape[1] - thresh):
            group = np.transpose(np.where(mask[i:i+thresh, j:j+thresh]))
            for k in range(len(group)):
                for l in range(k):
                    try:
                        u.unite(pixelMap[tuple(np.array([i,j]) + group[k])], pixelMap[tuple(np.array([i,j]) + group[l])])
                    except:
                        pdb.set_trace()
    groups = u.groups()
    
    clusters = [[pixels[i,:] for i in group] for group in groups]
    return clusters

@jit
# Aaron: For simplicity remove stride
def compute_convolution(I, T):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)
    
    ### Aaron: Added code for computing convolution ###
    kernel = T.astype(np.float32)
    K = kernel.flatten()
    K /= np.sqrt(np.sum(kernel * kernel)) # Normalize both kernel and ultimately image patch
    
    paddedIm = np.ones((I.shape[0] + kernel.shape[0]-1, I.shape[1] + kernel.shape[1]-1, 3))
    padSize = (int((kernel.shape[0]-1)/2), int((kernel.shape[1]-1)/2))
    for i in range(3):
        paddedIm[:,:,i] *= np.mean(I[:,:,i])
        paddedIm[padSize[0]:I.shape[0] + padSize[0], padSize[1]:I.shape[1] + padSize[1], :] = I
    
    # Now, do sliding window
    response = np.zeros((paddedIm.shape[0] - kernel.shape[0] + 1, paddedIm.shape[1] - kernel.shape[1] + 1))
    
    for i in range(response.shape[0]):
        for j in range(response.shape[1]):
            imPatch = (paddedIm[i:i+kernel.shape[0],j:j+kernel.shape[1],:]).astype(np.float32).flatten()
            imPatch /= np.sqrt(np.sum(imPatch * imPatch))
            response[i,j] = np.sum(imPatch * K)
    
    assert response.shape[0] == n_rows
    assert response.shape[1] == n_cols
        
    return response

def predict_boxes(I, heatmap, ACCEPT):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''
    output = []
    
    pixels, redMask = findRed(I, False)

    # Consider the intersection of where heatmap is red and also sufficiently
    # close to template
    mask = redMask * (heatmap > ACCEPT)
            
    pixels, edgeMask = findEdges(mask)

    # 3 is cluster dist threshold
    clusters = clusterPixels(pixels, edgeMask, 3) 

    # Remove clusters which are too small or too large
    clusters = [cluster for cluster in clusters if len(cluster) > 5 and len(cluster) < 120]
        
    im = Image.fromarray((heatmap * 255).astype(np.uint8))

    costs = []
    circles = []
    for cluster in clusters:
        pixels = np.array(cluster) # Nx2 now
        # Use center of mass as the center, look at how much distance of points
        # from center fluctuates, if too high then reject
        cm = np.mean(pixels, axis=0)
        radii = np.linalg.norm(pixels - cm, axis=1)
        r = np.mean(radii)
        cost = np.std(radii) / r # Fractional uncertainty/error
        
        costs.append(cost)
        
        y, x = cm
        circles.append([x, y, r])
        draw = ImageDraw.Draw(im)
        draw.point([(x-2, y), (x-1, y), (x, y), (x+1, y), (x+2, y), (x, y-2), (x, y-1), (x, y+1), (x, y+2)])
        draw.ellipse((x-r, y-r, x+r, y+r))
    
        # Use fixed aspect ratio for traffic light
        xK, yK, rK = (13, 13, 7)
        hK, wK = 28, 28 # Use a square bounding box to match annotations
        scale = r / rK    
        xK = int(scale * xK)
        yK = int(scale * yK)
        hK = int(scale * hK)
        wK = int(scale * wK)
        start = (int(y - yK), int(x - xK))
        
        # Need to account for partially occluded traffic light
        tl_row = max(start[0], 0)
        tl_col = max(start[1], 0)
        br_row = min(start[0]+ hK, I.shape[0])
        br_col = min(start[1]+ wK, I.shape[1])
        
        # Use heuristic combination of highest contained response value and 
        # circle cost to dictate confidence
        # Response value ~ 0.8 and cost ~ 0.3 to give rough sense
        score = np.clip(np.max(heatmap[tl_row:br_row, tl_col:br_col]) - 1/2 * cost, 0, 1)

        box = [tl_row,tl_col,br_row,br_col,score]
        output.append(box)
        
        draw = ImageDraw.Draw(im)
        y0, x0, y1, x1, _ = box
        draw.rectangle([x0, y0, x1, y1])
        
        fnt = ImageFont.truetype("arial.ttf", 10)
    
        draw.text((max(x0-15,0),max(y0-15, 0)), str(score)[:3], font=fnt, fill=0)
    
    return output, np.array(im), redMask, edgeMask

def detect_red_light_mf(I, name, disp=False):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    # Visualize the initial image
    if disp:
        plt.figure()
        plt.imshow(I)
    
    # Create a new folder
    fname = preds_path + '/' + name
    os.makedirs(fname, exist_ok=True)
    
    kernels = loadTemplates()
        
    heatmaps = []
    for kernel in kernels:
        heatmaps.append(compute_convolution(I, kernel))
    heatmap = np.max(heatmaps, axis=0) # maxpool
        
    # Visualize the output response
    if disp:
        plt.figure()
        plt.imshow(heatmap, cmap='gray')
       
    Image.fromarray((heatmap * 255).astype(np.uint8)).save(fname + '/heatmap.jpg', quality=95)
    output, visualizedIm, redMask, edgeMask = predict_boxes(I, heatmap, ACCEPT)

    Image.fromarray(visualizedIm).save(fname + '/boxedHeatmap.jpg')
    
    Image.fromarray(visualize(I, output).astype(np.uint8)).save(fname + '/ogBoxed.jpg', quality=95)
    
    Image.fromarray((redMask * 255).astype(np.uint8)).save(fname + '/redMask.jpg', quality=95)
    
    Image.fromarray((edgeMask * 255).astype(np.uint8)).save(fname + '/edgeMask.jpg', quality=95)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# load splits: 
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

#### TODO: REMOVE ####
# Used RL-155, RL-259, RL-062 traffic light templates
# file_names_train = ['RL-062.jpg']
# file_names_train = [] # Temporary so skip right to test

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    if i % 10 == 0:
        print('Train progress = ' + str(i/len(file_names_train)))
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I, file_names_train[i][:-4])

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):
        if i % 10 == 0:
            print('Test progress = ' + str(i/len(file_names_test)))
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I, file_names_test[i][:-4])

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
