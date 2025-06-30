from external import *

from caching import *
from data_handling import *

from data.sat2graph.decoder import * 
from data.sat2graph.common import * 

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

# Tensorflow information (updated by functions).
tf_state = {
    "is_initiated": False
}

# Locations of files and folders for a place.
def sat_locations(place):
    return  {
        # Pre-requisites.
        "image": get_cache_file(f"sat/{place}.png"),
        "pbmodel": get_cache_file(f"sat/sat2graph/globalv2.pb"),
        # Intermediate data files.
        "partial_gtes": get_cache_file(f"sat/sat2graph/partial-gtes/{place}"),
        "intermediate": get_cache_file(f"sat/sat2graph"),
        # Resulting files.
        "result": get_data_file(f"data/graphs/sat-{place}.graph"),
        "image-results": get_data_file(f"data/images/sat/{place}")
    }
def partial_gte_location(place, counter):
    return get_cache_file(f"sat/sat2graph/partial-gtes/{place}/{counter}.pkl")


def obtain_sat_graph(place):
    """End-to-end logic to infer sat graph of a place."""

    # Check prerequisite files exist.
    locations = sat_locations(place)
    if not path_exists(locations["image"]):
        raise Exception(f"Image for {place} does not exist.")
    if not path_exists(locations["pbmodel"]):
        raise Exception(f"GlobalV2 model does not exist.")

    # Load the model first
    if not tf_state.get("is_initiated", False):
        load_model()

    img, metadata = read_png(sat_locations(place)["image"])
    gte     = infer_gte(img, place)
    #todo: Fine-tune GTE by re-iterating keypoints (recover code 276052528c55553f178ff6ae209095229923809e)
    G   = decode_gte(gte, properties=decode_properties)
    write_pickle(f"{sat_locations(place)["intermediate"]}/graph-raw.pk")
    write_png(render_graph(G, height=height, width=width), f"{sat_locations(place)["image_results"]}/graph-raw.png")

    G   = optimize_graph(G)
    write_pickle(f"{sat_locations(place)["intermediate"]}/graph-refined.pk")
    write_png(render_graph(G, height=height, width=width), f"{sat_locations(place)["image_results"]}/graph-refined.png")

    G  = double_graph(G)
    write_graph(G, sat_locations(place)["result"])
    write_png(render_graph(graph, height=height, width=width, background=img, draw_intersection=False), f"{sat_locations(place)["image_results"]}/graph-with-background.png")
    

def load_model():
    """Load globalv2 model into memory. Updates global variable `tf_state`."""
    global tf_state
        
    print("Loading TF:")
    print("* GPU properties.")
    # GPU properties.
    gpu_options = tf.GPUOptions(allow_growth=True)
    tfcfg = tf.ConfigProto(gpu_options=gpu_options)
    tfcfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1 # enable xla 

    print("* Session.")
    # Model (tensorflow) setup.
    sess = tf.Session(config=tfcfg)
    pbfile_location = sat_locations("")["pbmodel"]

    print("* Optimized state.")
    #todo: if cannot find model at file location, raise error with message saying this.
    with tf.gfile.GFile(pbfile_location, 'rb') as f:
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
    tf_state["output"]     = tf.import_graph_def(graph_def_optimized, return_elements=['output:0'])[0]

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


def infer_partial_gte(image, road):
    
    tf_output     = tf_state["output"]    
    tf_inputsat   = tf_state["inputsat"]  
    tf_inputroad  = tf_state["inputroad"] 
    tf_istraining = tf_state["istraining"]
    sess          = tf_state["sess"]

    out = sess.run(tf_output, feed_dict={tf_inputsat: image, tf_inputroad: road, tf_istraining: False})
    return out[0]


def progress(x):
    """Printing progress."""
    n = int(x * 40)
    sys.stdout.write("\rProgress (%3.1f%%) "%(x*100.0) + ">" * n  + "-" * (40-n)  )
    sys.stdout.flush()


def pad_graph(graph, left=0, top=0):
    """Add offset to each graph position."""
    graph2 = {}

    for element, targets in graph.items():
        (y, x) = element
        graph2[(y + top, x + left)] = [(y + top, x + left) for (y, x) in targets]
    
    return graph2

def double_graph(graph):
    """Double all values in graph."""
    graph2 = {}

    for element, targets in graph.items():
        (y1, x1) = element
        graph2[(2 * y1, 2 * x1)] = [(2 * y, 2 * x) for (y, x) in targets]
    
    return graph2


def infer_gte(sat_img, place):
    """Infer GTE from a satellite image."""

    sat_height = sat_img.shape[0]
    sat_width  = sat_img.shape[1]
    # print("height:", sat_height)
    # print("width :", sat_width)

    # Inferrence parameters.
    image_size = 352       # Perceptive field of neural network.
    scale      = 2         # Scale perceptive field of satellite visual cognition part, higher quality/detail.
    stride     = 88        # Step after each inferrence (a quarter of the inferrence window size).
    feat_size  = 2 + 4 * 6 + 5 # Feature size, 6 edge probabilities and some additional for road type, but you don't understand why two necessary for vertexness. 

    # Expect input image to be a multiple of stride.
    check(sat_width  % stride == 0, expect="input image to be a multiple of stride")
    check(sat_height % stride == 0, expect="input image to be a multiple of stride")

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

    # Construct weights (for masking results, provide more value to center of inferrence result image.)
    # Note: Do not apply scale here, because inferrence provides same info.
    weights = np.zeros((image_size, image_size, feat_size)) + 0.00001  # Outer 32 pixels are ignored.
    a = stride // 3
    b = 2 * a
    c = stride
    weights[a:image_size-a, a:image_size-a, :] = 0.5 # Some relevance on the 32 to 56 pixel boundary.
    weights[b:image_size-b, b:image_size-b, :] = 1.0 # Even more on 56 to 88.
    weights[c:image_size-c, c:image_size-c, :] = 1.5 # Center part (half of the image) is emphasized with most importance, thus assumption to be inferred correctly.

    # For single inferrence.
    sat_batch       = np.zeros((1, image_size * scale, image_size * scale, 3)) # Part of satellite image to infer on.
    #help: What does road_batch do? Used to re-iterate on changes?
    road_batch      = np.zeros((1, image_size, image_size, 1)) # Road batch, note we dont scale since there is no additional information to store.

    # Iteration logic.
    total_counter = len(list(range(0, inf_height - image_size, stride))) * len(list(range(0, inf_width - image_size, stride)))
    counter = 0

    print("Inferring partial gte's.")
    for y in range(0, inf_height - image_size, stride):
        for x in range(0, inf_width - image_size, stride):
            counter += 1
            print(f"{counter}/{total_counter}")

            # print((y+image_size) * scale - y * scale, (x+image_size) * scale - x * scale)
            if sat_img[0, y * scale:(y+image_size) * scale, x * scale:(x+image_size) * scale, :].shape != (image_size * scale, image_size * scale, 3):
                # Error:
                print(f"y: {y}, x: {x}")
                print(sat_img[0, y * scale:(y+image_size) * scale, x * scale:(x+image_size) * scale, :].shape)
                breakpoint()
            sat_batch[0,:,:,:] = sat_img[0, y * scale:(y+image_size) * scale, x * scale:(x+image_size) * scale, :] 
            
            # Check file already written to disk.
            if path_exists(partial_gte_location(place, counter)): 
                continue

            partial_gte = infer_partial_gte(sat_batch, road_batch).reshape((image_size, image_size, feat_size))

            # Store partial gte on disk.
            write_pickle(partial_gte_location(place, counter), partial_gte)
    
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

            partial_gte = read_pickle(partial_gte_location(place, counter))

            output[y:y+image_size, x:x+image_size, :] += np.multiply(partial_gte[:,:,:], weights)
            mask  [y:y+image_size, x:x+image_size, :] += weights # Update mask with weights.

    gte = np.divide(output, mask) # Normalize by weight.
    gte = gte[a:inf_height-a,a:inf_width-a,:]# Drop padding.

    return gte


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
