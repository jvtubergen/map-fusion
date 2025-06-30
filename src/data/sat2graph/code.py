import logging
import json 
import os 
import os.path  
import math 
import gc
import numpy as np 
import tensorflow.compat.v1 as tf 
from time import time, strftime
from subprocess import Popen
import sys 
from douglasPeucker import simpilfyGraph, colorGraph
import requests
import pickle 
import PIL
from PIL import Image
from gmaps.lib import *
from decoder import * 

PIL.Image.MAX_IMAGE_PIXELS = 246913576 + 1

tf_state = {
    "is_initiated": False
}


# Infer satellite/road data.
def infer(sat, road, pbfile=None):
    global tf_state
    if not tf_state["is_initiated"]:
        
        print("Loading TF:")
        print("* GPU properties.")
        # GPU properties.
        gpu_options = tf.GPUOptions(allow_growth=True)
        tfcfg = tf.ConfigProto(gpu_options=gpu_options)
        tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1 # enable xla 

        print("* Session.")
        # Model (tensorflow) setup.
        sess = tf.Session(config=tfcfg)
        if pbfile == None:
            pbfile = "./models/globalv2.pb"

        print("* Optimized state.")
        with tf.gfile.GFile(pbfile, 'rb') as f:
            graph_def_optimized = tf.GraphDef()
            graph_def_optimized.ParseFromString(f.read())

        print("* Loading nodes.")
        for node in graph_def_optimized.node:            
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            
        
        print("* Loading Tensors.")
        tf_state["sess"]       = sess
        tf_state["output"]     = tf.import_graph_def(graph_def_optimized, return_elements=['output:0'])

        # print("* Listing Tensors:")
        # graph = tf.get_default_graph()
        # for op in graph.get_operations():
        #     print(f"  Operation: {op.name}")
        #     for tensor in op.outputs:
        #         print(f"   - Tensor: {tensor.name}")

        tf_state["inputsat"]   = tf.get_default_graph().get_tensor_by_name('inputsat:0')
        tf_state["inputroad"]  = tf.get_default_graph().get_tensor_by_name('inputroad:0')
        tf_state["istraining"] = tf.get_default_graph().get_tensor_by_name('istraining:0')

        tf_state["is_initiated"] = True
    
    tf_output     = tf_state["output"]    
    tf_inputsat   = tf_state["inputsat"]  
    tf_inputroad  = tf_state["inputroad"] 
    tf_istraining = tf_state["istraining"]
    sess          = tf_state["sess"]

    out = sess.run(tf_output, feed_dict={tf_inputsat: sat, tf_inputroad: road, tf_istraining: False})
    return out[0]


# Printing progres..
def progress(x):
	n = int(x * 40)
	sys.stdout.write("\rProgress (%3.1f%%) "%(x*100.0) + ">" * n  + "-" * (40-n)  )
	sys.stdout.flush()


def read_image(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    image = np.array(image)
    return image.astype(float)


def save_image(image, fname):
    Image.fromarray(image.astype(np.uint8)).save(fname)


# Example (Retrieve/Construct image with GSD ~0.88 between two coordinates):
def run_example():
    upperleft  = (41.799575, -87.606117)
    lowerright = (41.787669, -87.585498)
    scale = 1
    zoom = 17 # For given latitude and scale results in gsd of ~ 0.88
    api_key = read_api_key()
    # superimage = construct_image(upperleft, lowerright, zoom, scale, api_key)   # Same result as below.
    superimage, coordinates = construct_image(upperleft, lowerright, zoom-1, scale+1, api_key) # Same result as above.
    write_image(superimage, "superimage.png")


# Double all values in graph.
def double_graph(graph):
    graph2 = {}

    for element, targets in graph.items():
        (y1, x1) = element
        graph2[(2 * y1, 2 * x1)] = [(2 * y, 2 * x) for (y, x) in targets]
    
    return graph2


# Add offset to each graph position.
def pad_graph(graph, left=0, top=0):
    graph2 = {}

    for element, targets in graph.items():
        (y, x) = element
        graph2[(y + top, x + left)] = [(y + top, x + left) for (y, x) in targets]
    
    return graph2


# Infer GTE from a satellite image. Globalv2 uses scale of 2 and cityv1 uses a scale of 1.
def infer_gte(sat_img, scale=None, pbfile=None, tmpfolder=None):

    assert type(scale) == type(0)

    sat_height = sat_img.shape[0]
    sat_width  = sat_img.shape[1]
    print("height:", sat_height)
    print("width :", sat_width)

    # Inferrence parameters.
    image_size = 352       # Perceptive field of neural network.
    scale      = 2         # Scale perceptive field of satellite visual cognition part, higher quality/detail.
    stride     = 88        # Step after each inferrence (a quarter of the inferrence window size).
    feat_size  = 2 + 4 * 6 + 5 # Feature size, 6 edge probabilities and some additional for road type, but you don't understand why two necessary for vertexness. 

    # Expect input image to be a multiple of stride.
    assert sat_width  % stride == 0 
    assert sat_height % stride == 0 

    # Decode parameters.
    v_thr     = 0.05
    e_thr     = 0.01
    snap_dist = 15
    snap_w    = 100

    # Add a third of stride to image.
    a = (stride // 3) * scale
    sat_img = np.pad(sat_img, ((a,a),(a,a),(0,0)), 'constant') # Pad sat image with 32 pixels on each side.

    # Normalize image to [-0.5, 0.5] and format for usage with model.
    max_v = 255
    sat_img = (sat_img / max_v - 0.5) * 0.9 
    sat_img = sat_img.reshape((1, sat_height + 2*a, sat_width + 2*a, 3))

    # Actual image to act on with inferrence.
    inf_width  = (sat_width + 2*a) // 2
    inf_height = (sat_height + 2*a) // 2

    # Construct weights.
    # Note: Do not apply scale here, because inferrence provides same info.
    # Weights for masking results (valueing center of inferrence result image).
    weights = np.zeros((image_size, image_size, feat_size)) + 0.00001  # Outer 32 pixels are ignored.
    a = stride // 3
    b = 2 * a
    c = stride
    weights[a:image_size-a, a:image_size-a, :] = 0.5 # Some relevance on the 32 to 56 pixel boundary.
    weights[b:image_size-b, b:image_size-b, :] = 1.0 # Even more on 56 to 88.
    weights[c:image_size-c, c:image_size-c, :] = 1.5 # Center part (half of the image) is emphasized with most importance, thus assumption to be inferred correctly.

    # For single inferrence.
    sat_batch       = np.zeros((1, image_size * scale, image_size * scale, 3)) # Part of satellite image to infer on.
    road_batch      = np.zeros((1, image_size, image_size, 1)) # Road batch, note we dont scale since there is no additional information to store.

    # Iteration logic.
    total_counter = len(list(range(0, inf_height - image_size, stride))) * len(list(range(0, inf_width - image_size, stride)))
    counter = 0

    if tmpfolder == None:
        partial_gtes = [] # Store partial gtes in a list.

    print("Inferring partial gte's.")
    for y in range(0, inf_height - image_size, stride):
        for x in range(0, inf_width - image_size, stride):
            counter += 1
            print(f"{counter}/{total_counter}")
            # print((y+image_size) * scale - y * scale, (x+image_size) * scale - x * scale)
            if sat_img[0, y * scale:(y+image_size) * scale, x * scale:(x+image_size) * scale, :].shape != (image_size * scale, image_size * scale, 3):
                print(f"y: {y}, x: {x}")
                print(sat_img[0, y * scale:(y+image_size) * scale, x * scale:(x+image_size) * scale, :].shape)
                breakpoint()
            sat_batch[0,:,:,:] = sat_img[0, y * scale:(y+image_size) * scale, x * scale:(x+image_size) * scale, :] 
            
            # Check file already written to disk.
            if tmpfolder != None and os.path.exists(f"{tmpfolder}/partial_gte{counter}.pkl"): 
                continue

            partial_gte = infer(sat_batch, road_batch, pbfile=pbfile).reshape((image_size, image_size, feat_size))
            if tmpfolder != None:
                # Store partial gte on disk.
                pickle.dump(partial_gte, open(f"{tmpfolder}/partial_gte{counter}.pkl", "wb"))
            else:
                partial_gtes.append(partial_gte)
    
    # Drop memory on satellite image.
    del sat_img
    gc.collect()
    
    # For accumulating inferrence results.
    mask            = np.zeros((inf_height, inf_width, feat_size)) + 0.00001 # Masking values used to normalize.
    output          = np.zeros((inf_height, inf_width, feat_size)) # Output GTE where inferrence is stored.
    
    print("Accumulating partial gte's into output and mask.")
    counter = 0
    for y in range(0, inf_height - image_size, stride):
        for x in range(0, inf_width - image_size, stride):
            counter += 1
            print(f"{counter}/{total_counter}")

            if tmpfolder != None:
                partial_gte = pickle.load(open(f"{tmpfolder}/partial_gte{counter}.pkl", "rb"))
            else:
                partial_gte = partial_gtes[counter - 1]

            output[y:y+image_size, x:x+image_size, :] += np.multiply(partial_gte[:,:,:], weights)
            mask  [y:y+image_size, x:x+image_size, :] += weights # Update mask with weights.

    gte = np.divide(output, mask) # Normalize by weight.
    gte = gte[a:inf_height-a,a:inf_width-a,:]# Drop outer edge.
    return gte

def GTE_to_graph(gte, decode_properties=None):

    assert decode_properties != None
    assert type(decode_properties) == type({})

    properties = decode_properties
    
    # Step-1: Find vertices.
    vertexness = gte[:,:,0]
    keypoints = np.copy(vertexness)
    smooth_kp = scipy.ndimage.filters.gaussian_filter(keypoints, 1)
    strongest_keypoint = np.amax(smooth_kp)
    if strongest_keypoint < 0.001:
        raise Exception("Keypoints are basically zero, there is something off in the graph tensor.")
    smooth_kp = smooth_kp / strongest_keypoint # Normalize.
    keypoints = detect_local_minima(-smooth_kp, smooth_kp, properties["vertex_threshold"])
    breakpoint()

    cc = 0 

    # Locate endpoints (we do this becasue the local maxima may not represent all the vertices).
    edgeEndpointMap = np.zeros(gte.shape)
    for i in range(len(keypoints[0])):
        if cc > kp_limit:
            break 
        cc += 1

        x,y = keypoints[0][i], keypoints[1][i]

        for j in range(max_degree):

            if imagegraph[x,y,2+4*j] * imagegraph[x,y,0] > thr * thr: # or thr < 0.2:
            
                x1 = int(x + vector_norm * imagegraph[x,y,2+4*j+2])
                y1 = int(y + vector_norm * imagegraph[x,y,2+4*j+3])

                if x1 >= 0 and x1 < w and y1 >= 0 and y1 < h:
                    edgeEndpointMap[x1,y1] = imagegraph[x,y,2+4*j] * imagegraph[x,y,0]

    edgeEndpointMap = scipy.ndimage.filters.gaussian_filter(edgeEndpointMap, 3)
    edgeEndpoints = detect_local_minima(-edgeEndpointMap, edgeEndpointMap, thr*thr*thr)


def gte_vertexness_to_image(gte):

    # Convert GTE vertexness to grayscale image.
    vertexness = gte[:,:,0]
    gray = 255 - vertexness * 255 # Scale to 255.
    image = np.repeat(gray[:, :, np.newaxis], repeats=3, axis=2) # Scale one channel to three channel.
    # image[:,0:100,:] = 255 # Set left to right to white.
    # image[0:100,:,:] = 0   # Set top to bottom to black.

    return image


def render_graph(graph, height=None, width=None, background=None, draw_intersection=True):

    color_node = (255,0,0)
    edge_width = 2

    if type(background) != type(None):
        color_edge = (0,255,255) # yellow
        image = 0.75 * background.copy()
    else:
        color_edge = (0,0,0) # black
        image = np.ones((height, width, 3)) * 255
        image = image.astype(float)

    # Draw edges.
    for k,v in graph.items():
        n1 = k 
        for n2 in v:
            cv2.line(image, (n1[1], n1[0]), (n2[1], n2[0]), color_edge, edge_width)

    # Draw nodes.
    scale = 1
    for k,v in graph.items():
        n1 = k 
        cv2.circle(image, (int(n1[1]) * scale, int(n1[0]) * scale), 2, (255,0,0),-1)
        
    if draw_intersection == True:

        cp, _ = locate_stacking_road(graph)
        for k, v in cp.items():
            e1 = k[0]
            e2 = k[1]
            cv2.line(image, (int(e1[0][1]),int(e1[0][0])), (int(e1[1][1]),int(e1[1][0])), (0,255,0),edge_width)
            cv2.line(image, (int(e2[0][1]),int(e2[0][0])), (int(e2[1][1]),int(e2[1][0])), (0,0,255),edge_width)
    
    return image


def workflow_extract_vertexness_from_image():
    input_file = sys.argv[1]
    print("Reading image.")
    sat_img = read_image(input_file)
    print("Inferring GTE.")
    gte = infer_gte(sat_img, scale=2)
    print("Save vertexness as image.")
    image = gte_vertexness_to_image(gte)
    save_image(image, "gte_vertexness.png")


# Infer graph from satellite image and save intermediate results.
def workflow_infer_and_save():

    # Decode and visualization parameters.
    decode_properties = {
        "vertex_threshold": 0.05,
        "edge_threshold"  : 0.01,
        "snap_distance"   : 15,
        "snap_weight"     : 100,
        "snap"            : True,
        "drop"            : True,
        "refine"          : True,
        "max_edge_degree" : 6
    }

    input_file = sys.argv[1]
    pbfile = sys.argv[2]
    tmpfolder = sys.argv[3]
    sat_img = read_image(input_file)
    
    # Inferring graph.
    gte = infer_gte(sat_img, scale=2, pbfile=pbfile, tmpfolder=tmpfolder)
    pickle.dump(gte, open("gte.pk", "wb"))
    gte = pickle.load(open("gte.pk", "rb"))

    # Decoding graph.
    # gte = pickle.load(open("gte.pk", "rb"))
    graph = decode_gte(gte, properties=decode_properties)
    pickle.dump(graph, open("graph-raw.pk", "wb"))
    save_image(render_graph(graph, height=height, width=width), "graph-raw.png")

    # Post-process graph.
    # graph = pickle.load(open("graph-raw.pk", "rb"))
    graph = optimize_graph(graph)
    pickle.dump(graph, open("graph-refined.pk", "wb"))
    save_image(render_graph(graph, height=height, width=width), "graph-refined.png") 

    graph  = double_graph(graph)
    pickle.dump(graph, open("graph.pk", "wb"))
    
    # Store graph alongside background image.
    # graph = pickle.load(open("graph.pk", "rb"))
    save_image(render_graph(graph, height=height, width=width, background=sat_img, draw_intersection=False), "graph-background.png")


workflow_infer_and_save()