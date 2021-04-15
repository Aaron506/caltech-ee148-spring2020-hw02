import os
import json
import numpy as np

# Aaron added
import pdb
from scipy.sparse.csgraph import maximum_flow, maximum_bipartite_matching
from scipy.sparse import csr_matrix
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def visualizePR(preds, gts, confidence_thrs):
    fig, ax = plt.subplots()
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)

    for iou in [0.25, 0.5, 0.75]:
        tp = np.zeros(len(confidence_thrs))
        fp = np.zeros(len(confidence_thrs))
        fn = np.zeros(len(confidence_thrs))
        for i, thr in enumerate(confidence_thrs):
            if i % 50 == 0: 
                print('Progress = ' + str(i / len(confidence_thrs)))
            tp[i], fp[i], fn[i] = compute_counts(preds, gts, iou_thr=iou, conf_thr=thr)
        
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)
        
        precision = np.divide(tp, tp + fp)
        recall = np.divide(tp, tp + fn)
        ax.plot(recall, precision, label='IOU_thresh = ' + str(iou))
    ax.legend()

    return fig, ax

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    [y0_1,x0_1,y1_1,x1_1] = box_1
    [y0_2,x0_2,y1_2,x1_2] = box_2
    
    ymax1 = max(y0_1, y1_1)
    ymin1 = min(y0_1, y1_1)
    xmax1 = max(x0_1, x1_1)
    xmin1 = min(x0_1, x1_1)
    ymax2 = max(y0_2, y1_2)
    ymin2 = min(y0_2, y1_2)
    xmax2 = max(x0_2, x1_2)
    xmin2 = min(x0_2, x1_2)
    
    # No intersection if either don't overlap in x or y
    intersection = max(0, (min(ymax1, ymax2) - max(ymin1, ymin2))) * max(0, (min(xmax1, xmax2) - max(xmin1, xmin2)))     
    union = (ymax1 - ymin1) * (xmax1 - xmin1) + (ymax2 - ymin2) * (xmax2 - xmin2) - intersection
        
    iou = intersection / union
    
    assert (iou >= 0) and (iou <= 1.0)
    
    return iou

def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5, disp=False):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0 # true positive
    FP = 0 # false positive
    FN = 0 # false negative
        
    # Loop over images
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        
        # Make a matrix for the max flow problem, row is ground truth
        # col is predicted box
        # Row 0 and row 1 are reserved for source and sink respectively
        # The next len(gt) rows are reserved for ground truth boxes
        # Then after that remaining rows are for pred boxes
        # n = len(gt) + len(pred) + 2 # +2 because add source and sink
        # A = np.zeros(n, n)
        A = np.zeros((len(gt), len(pred)))
        
        # Keep track of number of boxes that were not discarded
        keptBoxes = 0
        
        for i in range(len(gt)):
            for j in range(len(pred)):
                # Only add edges in the case where this bounding box
                # exceeds confidence threshold
                if pred[j][-1] > conf_thr:
                    keptBoxes += 1
                    iou = compute_iou(pred[j][:4], gt[i])
                    if iou > iou_thr:
                        # Set capacity of this edge to 1 to allow for a
                        # potential match
                        # Connect from ground truth to predicted box node
                        # A[i+2, len(gt)+2+j] = 1
                        # Connect all ground truth to source
                        # A[0, i+2] = 1
                        # Connect all predictions to sink
                        # A[len(gt)+2+j, 1] = 1
                        A[i, j] = 1
        A = csr_matrix(A)
        
        # Run Edmond-Karps to find max matching via max flow
        # res = maximum_flow(A, 0, 1)
        
        # correctPred = res.flow_value
        
        perm = maximum_bipartite_matching(A, perm_type='column')
        
        # The correct number of predictions is simply the number of locations
        # which are not -1 in perm
        truePositives = np.sum(np.array(perm) != -1)
        falsePositives = keptBoxes - truePositives
        falseNegatives = len(gt) - truePositives
        
        assert truePositives >= 0
        assert falsePositives >= 0
        assert falseNegatives >= 0
        
        TP += truePositives
        FP += falsePositives
        FN += falseNegatives
        
        if disp:
            im = Image.open(data_path + '/' + pred_file)
            for i, matchInd in enumerate(perm):
                if matchInd == -1: continue
                # i is the ground truth index
                # matchInd is the predicted both index
                draw = ImageDraw.Draw(im)
                color = (int(np.random.rand(1) * 255), \
                         int(np.random.rand(1) * 255), \
                         int(np.random.rand(1) * 255))
                try:
                    for box in [gt[i], pred[matchInd]]:
                        try:
                            (y0, x0, y1, x1) = tuple(box)[:4]
                        except:
                            pdb.set_trace()
                        draw.rectangle([x0, y0, x1, y1], outline=color)
                except:
                    print('hi')
                    pdb.set_trace()
    
            im.show()
        
        
    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = 'data/hw02_preds'
gts_path = 'data/hw02_annotations'
data_path = 'data/RedLights2011_Medium' # Aaron added for visualization

# load splits:
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.

### Aaron modified ###
confidence_thrs = []
for fname in preds_train:
    for box in preds_train[fname]:
        confidence_thrs.append(box[4]) # Add the score
        
# confidence_thrs = np.sort(confidence_thrs)

# The outlined approach of just using the confidence_thrs does not work well
# because it reflects the distribution of the confidences themselves.
# Instead simply rely on the max and min and do uniform spacing
confidence_thrs = np.linspace(np.min(confidence_thrs), np.max(confidence_thrs), 100)

######
# confidence_thrs = np.sort(np.array([preds_train[fname] for fname in preds_train]),dtype=float) # using (ascending) list of confidence scores as thresholds

fig, ax = visualizePR(preds_train, gts_train, confidence_thrs)
ax.set_title('Training PR Curve', fontsize=14)

# tp_train = np.zeros(len(confidence_thrs))
# fp_train = np.zeros(len(confidence_thrs))
# fn_train = np.zeros(len(confidence_thrs))
# for i, thr in enumerate(confidence_thrs):
#     if i % 50 == 0: 
#         print('Train progress = ' + str(i / len(confidence_thrs)))
#     tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=thr)

# tp_train = np.array(tp_train)
# fp_train = np.array(fp_train)
# fn_train = np.array(fn_train)

# # Plot training set PR curves

# precision = np.divide(tp_train, tp_train + fp_train)
# recall = np.divide(tp_train, tp_train + fn_train)

# plt.figure()
# plt.title('PR Curve for train set')
# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')

if done_tweaking:
    print('Code for plotting test set PR curves.')  
    fig2, ax2 = visualizePR(preds_test, gts_test, confidence_thrs)
    ax2.set_title('Test PR Curve', fontsize=14)
    

### Aaron added plotting code ###


######