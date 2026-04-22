from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, send, emit
from DrawData import GraphHandler
from Model import LineTrainer, PatternTrainer
from line import Line
import json
import os
import os.path as osp
from os import listdir
from config import config
import random
import numpy as np
from pathlib import Path
import torch
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Configure Socket.IO with proper timeout settings to prevent connection drops
# ping_timeout: how long to wait for pong response (in seconds)
# ping_interval: how often to send ping (in seconds)
# max_http_buffer_size: maximum size of messages (in bytes)
socketio = SocketIO(
    app,
    ping_timeout=60,  # Increased from default 20 to handle long-running operations
    ping_interval=25,  # Default ping interval
    max_http_buffer_size=1e8,  # 100MB to handle large messages
    cors_allowed_origins="*",
    async_mode='threading'  # Use threading mode to prevent blocking
)

#create folders if they dont exist yet
Path("./baseData").mkdir(exist_ok=True)
Path("./lineModels").mkdir(exist_ok=True)
Path("./saved_svgs").mkdir(exist_ok=True)

#create folders if they dont exist yet
Path("./data").mkdir(exist_ok=True)

#delete all content in data folder
data_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
for file in os.listdir(data_path):
    file_path = osp.join(data_path, file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            import shutil
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')



gh = GraphHandler()
init_pattern = True
line_deposit = []

pattern_trainers = {
    "triangles": PatternTrainer("triangles"),
    "boxes": PatternTrainer("boxes"),
    "swirls": PatternTrainer("swirls"),

}
line_trainers = {
    "triangles": LineTrainer("triangles"),
    "boxes": LineTrainer("boxes"),
    "swirls": LineTrainer("swirls"),
}
correction = False


def get_or_create_trainers(name):
    """Return LineTrainer and PatternTrainer for ``name``, loading from disk if needed.

    ``add:stamp`` / ``add:visual`` used to only see models pre-registered in these dicts
    (initially just ``grid``). Training creates trainers in-memory but did not insert them,
    so any other UI model name raised KeyError. Lazily construct and cache like other handlers.
    """
    if name not in line_trainers:
        line_trainers[name] = LineTrainer(name)
    if name not in pattern_trainers:
        pattern_trainers[name] = PatternTrainer(name)
    return line_trainers[name], pattern_trainers[name]


#path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'dataset-test')
#dataset = MyOwnDataset("testdata", path)






@socketio.event
def connect():
    print("User connected")
    emit('init', config)
    emit('correctionChanged', {'correction': correction})
    mlist = getModels()
    print(mlist)
    emit('models', mlist)
    referer = request.headers.get("Referer", "")
    if "/draw" in referer:
        info = {"lines": [line.to_JSON() for line in gh.lines if line is not None]}
        emit('draw:lines', info)
        if gh.line_trainer is None:
            change_model("swirls")
        emit("set:info", {"model": gh.line_trainer.name, "correction": correction})

@socketio.event
def disconnect():
    print("User disconnected")

@socketio.on_error_default
def default_error_handler(e):
    print(f"Socket.IO error: {e}")
    import traceback
    traceback.print_exc()

#@socketio.on('new line')
#def new_line(points):
#    print("new line received", points)
#    bd.createData(points)

@socketio.on('add:line')
def add_line(data):
    print("line received", data)
    gh.add_lines([data])
    info = {}
    info["lines"] = [line.to_JSON() for line in gh.lines]
    print("lines", info["lines"])

    emit('draw:lines', info)

@socketio.on('add:visual')
def add_visual(data):
    #print("visual received", data
    gh.lines = [line for line in gh.lines if line is not None]
    position = {'x': float(data['position']['x']), 'y': float(data['position']['y'])}

    line_data = data['line']
    line = Line(line_data['points'], line_data['scale'], line_data['rotation'], position=position)
    print("line", line.to_JSON())
    
    line = gh.add_missing_latent_vector_to_line(line)
    closest_original, distance, closest_idx = gh.get_closest_original_line(line)
    print("closest original", closest_original.to_JSON())
    
    print("distance", distance, closest_idx)
    print(line.latent_vectors, closest_original.latent_vectors)

    scale_offset = line.scale - closest_original.scale
    rotation_offset = line.rotation - closest_original.rotation
    print("scale offset", scale_offset)
    print("rotation offset", rotation_offset)

    new_line = closest_original.clone()
    new_line.position = position
    new_line.rotation = line.rotation
    new_line.scale = line.scale

    if correction:
        corrected_line = apply_correction(new_line, position)
        
        new_line.scale = new_line.scale + scale_offset
        new_line.rotation = new_line.rotation + rotation_offset
        new_line.position = position

    if line is not None:
        new_line.added_with_model = gh.line_trainer.name
        new_line.added_at_stage = "visual"
        if correction:
            new_line.added_at_stage = "visual correction"
        gh.lines.append(new_line)
        info = {}
        info["lines"] = [line.to_JSON() for line in gh.lines if line is not None]
        #print("lines", info["lines"])
        emit('draw:lines', info)
    else:
        print("no line predicted")

def apply_correction(line, position):
    prediction = gh.predict_to_draw(position)
    if prediction is not None:
        line.position = prediction.position
        line.rotation = prediction.rotation
        line.scale = prediction.scale
    return line
    
@socketio.on('change:model')
def change_model(data):
    print("change model received", data)
    name = data
    lineTrainer = line_trainers[name]
    patternTrainer = pattern_trainers[name]
    gh.original_lines = []
    gh.set_default_trainers(pattern_trainer=patternTrainer, line_trainer=lineTrainer)
    gh.calculate_original_lines()
    emit('modelChanged', {'name': name})

@socketio.on('change:correction')
def change_correction(data):
    global correction
    print("change correction received", data)
    correction = bool(data['correction'])
    emit('correctionChanged', {'correction': correction})

@socketio.on('add:stamp')
def add_stamp(data):
    print("stamp received")
    name = gh.line_trainer.name
    print("using name", name)
    gh.lines = [line for line in gh.lines if line is not None]
    position = {'x': float(data['position']['x']), 'y': float(data['position']['y'])}
    if len(gh.lines) == 0:
        
        #gh.calculate_original_lines()
        line =gh.original_lines[random.randint(0, len(gh.original_lines) - 1)]
        line.position = position
    else:
        gh.add_missing_latent_vectors()
        line = gh.predict_to_draw(position)
        if line is None:
            #gh.calculate_original_lines()
            line =gh.original_lines[random.randint(0, len(gh.original_lines) - 1)]
            line.position = position
        if not correction:
            print("not correction - resetting position to", position)
            line.position = position

    if line is not None:
        closest_original, distance, closest_idx = gh.get_closest_original_line(line)
        new_line = closest_original.clone()
        new_line.position = line.position
        if correction:
            new_line.rotation = line.rotation
            new_line.scale = line.scale

        new_line.added_with_model = name
        new_line.added_at_stage = "stamp"
        if correction:
            new_line.added_at_stage = "stamp correction"
        gh.lines.append(new_line)
        info = {}
        info["lines"] = [line.to_JSON() for line in gh.lines]
        emit('draw:lines', info)
    else:
        print("no line predicted")

@socketio.on('erase:lines')
def erase_lines(data):
    print("erase lines received", data)
    if isinstance(data, list):
        indices = data
    else:
        indices = data.get('indices', [])
    for index in indices:
        idx = int(index)
        if 0 <= idx < len(gh.lines):
            gh.lines[idx] = None
    gh.lines = [line for line in gh.lines if line is not None]
    info = {}
    info["lines"] = [line.to_JSON() for line in gh.lines]
    emit('draw:lines', info)

@socketio.on('clear')
def clear():
    print("clear received")
    gh.clear()
    info = {}
    info["lines"] = [line.to_JSON() for line in gh.lines]
    emit('draw:lines', info)

@socketio.on('undo')
def undo():
    print("undo received")
    gh.lines.pop()
    info = {}
    info["lines"] = [line.to_JSON() for line in gh.lines]
    emit('draw:lines', info)

@socketio.on('apply:noise')
def apply_noise(data):
    print("apply noise", data)
    noise_level = data['noise_level']
    if noise_level is None:
        noise_level = 0.01

    info = {}
    # gh.add_missing_latent_vectors()
    active_lines = [line for line in gh.lines if line is not None]
    noisy_lines = [None] * len(active_lines)
    batched_by_model = {}
    model_order = []

    for idx, line in enumerate(active_lines):
        model_name = line.added_with_model
        if model_name is None:
            noisy_lines[idx] = line.clone()
            continue

        if model_name not in batched_by_model:
            batched_by_model[model_name] = []
            model_order.append(model_name)
        batched_by_model[model_name].append((idx, line))

    for model_name in model_order:
        if model_name != gh.line_trainer.name:
            change_model(model_name)
        for idx, line in batched_by_model[model_name]:
            noisy_lines[idx] = gh.create_noisy_copy(
                line,
                noise_level=noise_level,
                latent_name=model_name
            )

    info["lines"] = [line.to_JSON() for line in noisy_lines]
    emit('draw:lines', info)

@socketio.on('deleteModel')
def delete_model(data):
    model_name = data['modelName']
    model_path = osp.join("./lineModels", model_name)
    model_path_basedata = osp.join("./baseData", model_name + "-line.pt")
    
    if osp.exists(model_path):
        try:
            os.remove(model_path)
            os.remove(model_path_basedata)
            print(f"Model {model_name} deleted successfully.")
            emit('modelDeleted', {'status': 'success', 'message': f'Model {model_name} deleted successfully.'})
        except Exception as e:
            print(f"Error deleting model {model_name}: {str(e)}")
            emit('modelDeleted', {'status': 'error', 'message': f'Error deleting model {model_name}: {str(e)}'})
    else:
        print(f"Model {model_name} not found.")
        emit('modelDeleted', {'status': 'error', 'message': f'Model {model_name} not found.'})

    # Update the list of available models
    mlist = getModels()
    emit('models', mlist)


@socketio.on('raw data')
def raw_data(data):
    gh.init_raw(data)
    print('raw data recieved')

@socketio.on('new dataset')
def new_dataset(data):
    #print("socket received:")
    #print(data)

    if len(data['list']) == 0:
        print("ERROR: no drawing data transmitted!")
    else:
        gh.init_lines(data['list'])
        gh.save_line_training_data(data['name'])

        lineTrainer = LineTrainer(data['name'])
        lineTrainer.trainModel(send_progress)
        
def send_progress(trainer, text, label=None):
   
    if isinstance(text, list):
        pointlist = []
        for z in text:
            print(z)
            points_tensor = trainer.decode_latent_vector(z)
            pointlist.append(Line(points_tensor).to_JSON())
        #print(pointlist)
        emit('progress', {'lines': pointlist} )
    else:
        print("sending progress", text)
        emit('progress', {'percent':text, 'label':label} )
        
def getLatentspaceLine(data):
    lineTrainer = LineTrainer(data['name'])
    x, edge_index = gh.create_line_graph(data['points'])
    z = lineTrainer.encodeLineVector(x, edge_index)
    line = lineTrainer.decode_latent_vector(z)

def getModels():
    onlyfiles = [f for f in listdir("./lineModels") if osp.isfile(osp.join("./lineModels", f))]
    return onlyfiles

@socketio.on('generate')
def generate(data):
    print(data)
    lineTrainer = LineTrainer(data['name'])

    lines1 = lineTrainer.generate(10, 'random')
    lines2 = lineTrainer.generate(10, 'random')
    
    lines = lines1 + lines2
    lines = list(map(lambda line: line.to_JSON(), lines)) 
    

    emit('result', {'list': lines})
    
@socketio.on('convertToLatentspace')
def convertToLatentspace(data):
    print(data)
    
    
    if data['name'] == "random":
        print("choosing RANDOM model")
        trainer = LineTrainer(random.choice(getModels()))
    else:
        trainer = LineTrainer(data['name'])
        
    pointlist = []    
    for line in data['list']:    

        x, edge_index = gh.create_line_graph(line['points'])
        z = trainer.encodeLineVector(x, edge_index)
        zMatch = trainer.getClosestMatch(z)
        
        
        latentLine = trainer.decode_latent_vector(zMatch)
        pointlist.append(GraphHandler.tensor2Points(latentLine))
    
    

    #for tensor in tensors:
    #    pointlist.append(tensor2Points(tensor))

    emit('latentLine', {'list': pointlist })
    
@socketio.on('compare')
def compare(data):
    trainer = LineTrainer(data['name'])
    
    pointlist = []
    originlist = []
    tensors, originpoints = trainer.extractOriginLineVectors()
    for z in tensors:
        print(z)
        tensor = trainer.decode_latent_vector(z)
        pointlist.append(GraphHandler.tensor2Points(tensor))
        
    for ori in originpoints:
        originlist.append(GraphHandler.tensor2Points(ori))
        
    emit('result', {'list': pointlist, "origins":originlist})


def prepare_sample_for_visualization(sample_data_list, lineTrainer, patternTrainer, predictions=None, data_jitter=0):
    info = {}
    info["ground_truth"] = []
    info["prediction"] = []
    info["sample_nodes"] = []
    info["dropped_out_nodes"] = []
    info["target_pos"] = []
    
    for idx, sample_data in enumerate(sample_data_list):
    
        if data_jitter > 0:
            sample_data.x = sample_data.x + torch.randn(sample_data.x.size()) * data_jitter
        sampled_lines = []

        for i in range(sample_data.x.size()[0]):
            line = GraphHandler.decompose_node_hidden_state(sample_data.x[i], lineTrainer)
            line.update_position_from_reference(sample_data.center_point, sample_data.max_dist)
            sampled_lines.append(line)

        ground_truth = GraphHandler.decompose_node_hidden_state(sample_data.y, lineTrainer)
        ground_truth.update_position_from_reference(sample_data.center_point, sample_data.max_dist)
        info["ground_truth"].append(ground_truth.to_JSON())

        dropped_out_nodes = []
        for dropped_id in sample_data.dropped_out_ids:
            line_pos = patternTrainer.template_data["lines"][dropped_id].position
            dropped_out_nodes.append(line_pos)

        if predictions is not None:
            prediction = predictions[idx]
            prediction_line = GraphHandler.decompose_node_hidden_state(prediction, lineTrainer)
            prediction_line.update_position_from_reference(sample_data.center_point, sample_data.max_dist)
            info["prediction"].append(prediction_line.to_JSON())

        info["target_pos"].append(GraphHandler.get_target_pos_from_sample_data(sample_data))

        info["sample_nodes"].extend([line.to_JSON() for line in sampled_lines])
        info["dropped_out_nodes"].extend(dropped_out_nodes)

    return info

def calculate_base_deltas(pattern_trainer, base_dataset, num_samples=50):
    """
    Calculate position deltas for base dataset samples where target_pos == ground_truth.
    Returns average and max delta norms.
    """
    from torch_geometric.loader import DataLoader
    
    # Limit to available samples
    num_samples = min(num_samples, len(base_dataset))
    
    # Batch the samples
    base_loader = DataLoader(base_dataset[0:num_samples], batch_size=num_samples, shuffle=False)
    base_batch = next(iter(base_loader))
    
    # Get predictions
    with torch.no_grad():
        pred = pattern_trainer.model.forward(base_batch.x, base_batch.edge_index, base_batch.batch, target_pos=base_batch.target_point)
        pred = pred.view(-1, 7)  # Reshape to [batch, 7]
        pred_pos = pred[:, 0:2]  # Extract positions
        
        # Get target positions
        target_pos = base_batch.target_point  # Already [batch, 2]
        
        # Calculate deltas
        deltas = pred_pos - target_pos
        delta_norms = torch.norm(deltas, dim=1)
        
        avg_delta = delta_norms.mean().item()
        max_delta = delta_norms.max().item()
        min_delta = delta_norms.min().item()
        
    return {
        'avg': avg_delta,
        'max': max_delta,
        'min': min_delta,
        'num_samples': num_samples
    }

@socketio.on('train pattern')
def new_pattern(data):
    lineTrainer = LineTrainer(data['name'])
    gh.clear()
    gh.set_default_trainers(line_trainer=lineTrainer)
    gh.init_lines(data['list'])
    gh.add_line_latentspace()
    gh.save_pattern_training_data(data['name'])

    pt = PatternTrainer(data['name'])
    gh.set_default_trainers(pattern_trainer=pt, line_trainer=lineTrainer)
    gh.load_template_from_pattern_trainer()

    info = {}
    info["base_list"] = [line.to_JSON() for line in gh.lines]
    emit('prediction', info)

    base_dataset = gh.calculate_base_dataset()
    noisy_dataset = gh.calculate_dataset_onthefly(nr_samples=len(base_dataset))

    validation_dataset = gh.calculate_dataset_onthefly(nr_samples=len(base_dataset)*0.05)
    pt.setup_from_test_sample(validation_dataset)
    
    threshold = 10
    count = 0
    
    
    data_jitter = 0
    dataset = base_dataset + noisy_dataset

    for i in range(1001):
        if count >= threshold:
            noisy_dataset = gh.calculate_dataset_onthefly(nr_samples=len(base_dataset))
            dataset = base_dataset + noisy_dataset
            #threshold -= 1
            count = 0
        #    if threshold <= 5:
        #        threshold = 5

        if i > 15:
            data_jitter = min(0.01, (i - 15) * 0.01 / 10)  # Ramp over 10 epochs
        pt.trainModel(dataset, data_jitter=data_jitter)
        count += 1

         # Check if learning rate has reached minimum
        current_lr = pt.optimizer.param_groups[0]['lr']
        min_lr = pt.scheduler.min_lrs[0] if pt.scheduler is not None else 0
        if current_lr < min_lr:
            print(f"Stopping training: Learning rate reached minimum ({current_lr})")
            break

        print("current_lr", current_lr, "min_lr", min_lr)
        visualize_dataset = gh.calculate_dataset_onthefly(nr_samples=3) + random.sample(base_dataset, 3)
        

        predictions = []
        for sample_data in visualize_dataset:
            prediction = pt.predict_from_sample(sample_data)
            predictions.append(prediction)

        info = prepare_sample_for_visualization(visualize_dataset, lineTrainer, pt, predictions=predictions, data_jitter=data_jitter)
        info["base_list"] = [line.to_JSON() for line in gh.lines]

        emit('prediction', info)



@socketio.on('inspect latent')
def inspect_latent(data):
    print("inspect latent", data)


    lineTrainer = LineTrainer(data['name'])
    pointlist = []
    latent_position_list = []
    tensors, _ = lineTrainer.extractOriginLineVectors()
    for z in tensors:
        print(z)
        tensor = lineTrainer.decode_latent_vector(z)
        pointlist.append(Line._tensor2Points(tensor))
        latent_position_list.append(z.tolist())


    # get the individual min and max value of each latent position direction
    if latent_position_list:
        # latent_position_list is a list of lists, e.g. [[x1, y1, z1], [x2, y2, z2], ...]
        # We want the min and max for each dimension
        import numpy as np
        latent_array = np.array(latent_position_list)
        min_latent = latent_array.min(axis=0)
        max_latent = latent_array.max(axis=0)
        print("Min value in each latent direction:", min_latent)
        print("Max value in each latent direction:", max_latent)

        dist = min(max_latent - min_latent) / 5
        print("Distance between min and max value in each latent direction:", dist, max_latent - min_latent)

    # Create a grid of points from min_latent to max_latent with spacing 'dist'
    # The grid will be in the latent space dimensions (usually 2D or 3D)
    grid_points = []
    if latent_position_list:
        # Determine the number of dimensions
        dims = len(min_latent)
        # For 2D or 3D latent spaces
        if dims == 2:
            x_vals = np.arange(min_latent[0], max_latent[0] + dist, dist)
            y_vals = np.arange(min_latent[1], max_latent[1] + dist, dist)
            for x in x_vals:
                for y in y_vals:
                    grid_points.append([x, y])
        elif dims == 3:
            x_vals = np.arange(min_latent[0], max_latent[0] + dist, dist)
            y_vals = np.arange(min_latent[1], max_latent[1] + dist, dist)
            z_vals = np.arange(min_latent[2], max_latent[2] + dist, dist)
            for x in x_vals:
                for y in y_vals:
                    for z in z_vals:
                        grid_points.append([x, y, z])
        else:
            print("ERROR: latent space is not 2D or 3D. Higher dimensions not implemented yet.")
        #print("Grid points in latent space:", grid_points)


    # For all grid_points, get the latent vector z from the lineTrainer and also get the points from the tensor.
    grid_pointlist = []
    grid_latent_position_list = []
    for grid_z in grid_points:
        # grid_z is a list (e.g. [x, y] or [x, y, z])
        # Convert to tensor if needed
        import torch
        z_tensor = torch.tensor(grid_z, dtype=torch.float32)
        # If the model expects a batch dimension, unsqueeze(0)
        # But from context, lineTrainer.decode_latent_vector(z) expects a 1D tensor
        tensor = lineTrainer.decode_latent_vector(z_tensor)
        grid_pointlist.append(Line._tensor2Points(tensor))
        grid_latent_position_list.append(grid_z)
    # Add to emit
    emit('latent', {
        'pointlist': pointlist + grid_pointlist,
        'latent_position_list': latent_position_list + grid_latent_position_list,
        'is_original': [True] * len(pointlist) + [False] * len(grid_pointlist)
        #'grid_pointlist': grid_pointlist,
        #'grid_latent_position_list': grid_latent_position_list
    })

@socketio.on('make noise')
def make_noise(data):
    print("make noise", data)
    noise_level = data['noise_level']
    if noise_level is None:
        noise_level = 0.01

    info = {}
    info["initial"] = [gh.create_noisy_copy(line,noise_level=noise_level).to_JSON() for line in gh.lines if line != None]
    emit('prediction', info)
    print("prediction emitted")



@socketio.on('sample pattern')
def sample_pattern(data):



    pt = PatternTrainer(data['name'])
    lineTrainer = LineTrainer(data['name'])
    gh.clear()
    gh.set_default_trainers(pattern_trainer=pt, line_trainer=lineTrainer)
    gh.load_template_from_pattern_trainer()

    sample_data = gh.calculate_dataset_onthefly(nr_samples=1)[0]
    print("sample_data", sample_data)

    info = prepare_sample_for_visualization(sample_data, lineTrainer, pt)
    info["base_list"] = [line.to_JSON() for line in gh.lines]

    emit('prediction', info)


@socketio.on('generate pattern')
def generate_pattern(data):
    """ pt = PatternTrainer(data['name'])
    lineTrainer = LineTrainer(data['name'])
    lines = []

    for i in range(10):
        z = lineTrainer.randomInitPoint()
        line = GraphHandler.decompose_node_hidden_state(z)
        lines.append(prediction2obj(line, lineTrainer))
        
    emit('prediction', {'base_list': lines}) """

    global init_pattern, line_deposit, all_line_lists

    if(init_pattern):
        init_pattern = False
        lineTrainer = LineTrainer(data['name'])
        pt = PatternTrainer(data['name'])
        gh.clear()
        gh.set_default_trainers(pattern_trainer=pt, line_trainer=lineTrainer)
        
        gh.calculate_original_lines()
        gh.calculate_line_thresholds()
        patch_data = gh.init_original(noise_level=0)
        

        #gh.random_fill()
        

        info = {}
        info["initial"] = [line.to_JSON() for line in gh.lines]
        info["patch_data"] = patch_data
        emit('prediction', info)
        print("prediction emitted")
       

        #flow_data = gh.calculate_flow_grid(grid_resolution=50)
        #info["flow_data"] = flow_data

        # INSERT_YOUR_CODE
        from collections import defaultdict

        # First, cluster by patch_id
        patch_clusters = defaultdict(list)
        for line in gh.lines:
            if not line.immutable:
                patch_id = getattr(line, "patch_id", None)
                patch_clusters[patch_id].append(line)

        # Now, within each patch_id, cluster by outside_directions
        clustered = {}
        for patch_id, lines in patch_clusters.items():
            dir_clusters = defaultdict(list)
            for line in lines:
                direction = getattr(line, "outside_directions", None)
                # Convert dict to hashable tuple if it's a dict
                if isinstance(direction, dict):
                    direction = tuple(sorted(direction.items()))
                dir_clusters[direction].append(line)
            clustered[patch_id] = dict(dir_clusters)

        # Make a list of all the line lists so it's easier to iterate over
        # Split each group into chunks of max length 3
        
        all_line_lists = []
        for patch_id, dir_dict in clustered.items():
            for direction, lines in dir_dict.items():
                # Append lines in chunks of 5
                #for i in range(0, len(lines), 5):
                #    all_line_lists.append(lines[i:i+5])
                all_line_lists.append(lines)
        all_line_lists.append([])

        # Iterate backwards to avoid skipping elements when removing items
        for i in range(len(gh.lines) - 1, -1, -1):
            line = gh.lines[i]
            if not line.immutable:
                gh.lines.pop(i)



        #gh.lines.extend(all_line_lists.pop(1))
        
        #print("lines for happyness check", len([line for line in gh.lines if not getattr(line, "immutable", False)]))
        #happyness_check(gh)
                   
        #info = {}
        #info["initial"] = [line.to_JSON() for line in gh.lines]
        #emit('prediction', info)
        

    else:
        info = {}
        try_later = []
        gen_state = "init"
        change_in_run = False
        backup_lines = []
        bonus_lines = []
        check_unknown = False
        last_run = 0

        second_stage_lines = []

        for l in [l for l in gh.lines if l.immutable]:
            l.added_at_stage = "patch"

        for i in range(1000000000):

            if len(all_line_lists) > 0:
                #try_later.extend([line for line in gh.lines if not line.immutable])
                gh.lines = [line for line in gh.lines if getattr(line, "immutable", False)]
                gh.lines.extend(all_line_lists.pop(0))
                if len(all_line_lists) == 0:
                    print("sleep to catch up")
                    #time.sleep(10)
            else:
                gen_state = "weave"
                if len(try_later) > 0:
                    gh.lines = [line for line in gh.lines if line is not None]
                    choice = random.choice(try_later)
                    #backup_lines.append(choice.clone())
                    try_later.remove(choice)
                    #backup_lines.append(choice.clone())
                    gh.lines.append(choice.clone())
                else:
                    if len(backup_lines) > 0:
                        try_later = [line.clone() for line in backup_lines]
                        backup_lines = []

                        print("check lastrun", last_run, len(try_later))
                        time.sleep(3)
                        if last_run == len(try_later):
                            check_unknown = True
                            print("CHECK UNKNOWN ENTERED")

                        last_run = len(try_later)
                        
                    else:
                        print("no backup lines")
                        info = {}
                        info["initial"] = [line.to_JSON() for line in gh.lines if line != None]        
                        emit('prediction', info)
                        exit()

            for line in [l for l in gh.lines if l is not None and not getattr(l, "immutable", False)]:
                
                line_idx = gh.lines.index(line)
                time_sleep = 0.1
                relax_value = 0.25

                
                
                if gen_state == "init":
                    if line is not None:

                        line.stopped = False
                        line.is_fixed = False
                        
                        predictions, average_lines = gh.evaluate_ensemble(line, gh.pattern_trainer.max_dist, distance_list=[0.01])
                        if len(average_lines) <= 0:
                            line.immutable = True
                            line.stopped = True
                            line.is_fixed = True
                            line.added_at_stage = "patch accept"
                            continue
                        average_line = average_lines[0]

                    
                    
                        are_similar, _ = average_line.are_similar(line, relaxation=relax_value)
                        if average_line is None or are_similar:
                            
                            line.immutable = True
                            line.stopped = True
                            line.is_fixed = True
                            line.added_at_stage = "patch accept"
                        else:
                            
                            #if test:
                                #bonus_lines.append(line)
                                #print("BONUS LINES", len(bonus_lines))

                            try_later.append(line)
                            line.stopped = True
                            line.is_fixed = True
                            

                        # clusters_list = predictions  # predictions contains clusters list from evaluate_ensemble
                        # predictions = []
                        # for cluster_number, cluster_lines in enumerate(clusters_list):
                        #     # Ensure cluster_lines is a list, not a single Line object
                        #     if not isinstance(cluster_lines, list):
                        #         cluster_lines = [cluster_lines]
                        #     for line in cluster_lines:
                        #         line.cluster_number = int(cluster_number)
                        #         predictions.append(line)
                        # if average_line is not None:         
                        #     predictions.append(average_line) 
                        #info["initial"] = [line.to_JSON() for line in gh.lines if line != None]
                        #info["ghost_lines"] = [line.to_JSON() for line in predictions]
                        #info["average_line"] = [b.to_JSON() for b in try_later]
                        #emit('prediction', info)
                        #time.sleep(1)
                        

                if gen_state == "weave":
                    line.stopped = False
                    line.is_fixed = False
                    info["average_line"] = []

                    _, average_lines = gh.evaluate_ensemble(line, gh.pattern_trainer.max_dist, distance_list=[0.01])
                    if len(average_lines) <= 0:
                        continue
                    average_line = average_lines[0]
                    test, error = average_line.are_similar(line, relaxation=relax_value)

                    print("test", test)
                    print("error", error)
                    print("lines lieft", len(try_later))

                    if test:
                        line.immutable = True
                        line.stopped = True
                        line.is_fixed = True
                        line.added_at_stage = "weave"
                        gh.lines.append(line.clone())
                    else:
                        if check_unknown:
                            _, checkup_lines = gh.evaluate_ensemble(average_line, gh.pattern_trainer.max_dist, distance_list=[0.01])
                            if len(checkup_lines) <= 0:
                                continue
                            checkup_line = checkup_lines[0]
                            checkup_test, error = checkup_line.are_similar(average_line, relaxation=relax_value)
                            closest_original, distance, closest_idx = gh.get_closest_original_line(average_line)
                            if checkup_test and distance < 0.3:
                                average_line.immutable = True
                                average_line.stopped = True
                                average_line.is_fixed = True
                                average_line.added_at_stage = "weave adjust"
                                gh.lines.append(average_line)
                                
                                info["initial"] = [line.to_JSON() for line in gh.lines if line != None]
                            
                                info["average_line"] = [average_line.to_JSON(), checkup_line.to_JSON()]
                                
                                #info["ghost_lines"] = [line.to_JSON() for line, _ in ranked_predictions]
                                
                                emit('prediction', info)
                            
                        else:
                            backup_lines.append(line.clone())

                    # r = random.random()/10
                    # clusters, predictions = gh.evaluate_ensemble(line, gh.pattern_trainer.max_dist, distance_list=[0,0.1+r,0.2+r,0.3+r,0.4+r,0.5+r,1+r])

                    
            

                    
                    # if len(predictions) <= 0:
                    #     gh.lines[line_idx] = None
                    #     continue

                    # max_distance = 0.7
                    # ranked_predictions = []
                    # for prediction in predictions:
                    #     similar_original, distance, closest_idx = gh.get_closest_original_line(prediction)
                    #     distance_value = distance.item() if isinstance(distance, torch.Tensor) else float(distance)
                    #     if distance_value <= max_distance:
                    #         ranked_predictions.append((prediction, distance_value))

                    # ranked_predictions.sort(key=lambda item: item[1])
                    # info["average_line"] = []
                    # if len(ranked_predictions) > 0:
                    #     best_prediction = ranked_predictions[0][0]

                    #     print("diff", best_prediction.latent_line_diff(line))
                    #     print("distance", distance_value)
                    #     print("lines lieft", len(try_later))

                    #     if best_prediction.latent_line_diff(line) > 1:

                    #         if check_unknown:
                    #             _, average_lines = gh.evaluate_ensemble(best_prediction, gh.pattern_trainer.max_dist, distance_list=[0.01])
                    #             average_line = average_lines[0]

                    #             test, _ = average_line.are_similar(best_prediction, relaxation=0)
                    #             if test:
                    #                 best_prediction.immutable = True
                    #                 best_prediction.stopped = True
                    #                 best_prediction.is_fixed = True
                    #                 best_prediction.added_at_stage = "weave adjust"
                    #                 gh.lines.append(best_prediction)
                    #             info["average_line"] = [best_prediction.to_JSON(), average_line.to_JSON()]
                    #             print("CHECK UNKNOWN", distance_value)
                    #         else:
                    #             backup_lines.append(line.clone())

                    #     else:

                    #         line.immutable = True
                    #         line.stopped = True
                    #         line.is_fixed = True
                    #         line.added_at_stage = "weave"
                    #         gh.lines.append(line.clone())
                    #         info["average_line"] = [line.to_JSON()]
                        
                    #info["initial"] = [line.to_JSON() for line in gh.lines if line != None]
                    #info["ghost_lines"] = [average_line.to_JSON()]
                    #info["average_line"] = [average_line.to_JSON()]
                    
                    #info["ghost_lines"] = [line.to_JSON() for line, _ in ranked_predictions]
                    
                    #emit('prediction', info)
                    #time.sleep(3)
                   
                    # best_prediction = None
                    # best_distance = float("inf")
                    # for prediction in predictions:
                    #     similar_original, distance, closest_idx= gh.get_closest_original_line(prediction)
                    #     #distance = similar_original.latent_line_diff(prediction)
                    #     print("distance", distance)
                    #     if distance < best_distance and distance < 0.25:
                    #         best_prediction = prediction
                    #         best_distance = distance

                    # if best_prediction is not None:
                    #     test_cluster, test_avg = gh.evaluate_ensemble(best_prediction, gh.pattern_trainer.max_dist, distance_list=[0.01])
                    #     print("TEST CLUSTER", len(test_cluster))
                    #     print("TEST CLUSTER distances", [line.pos_diff(best_prediction) for line in test_avg])
                    #     best_prediction.immutable = True
                    #     best_prediction.stopped = True
                    #     best_prediction.is_fixed = True
                    #     gh.lines.append(best_prediction)
                    #     print(f"ACCEPTED (distance: {best_distance})")

                    

                    
                    

                    
                    
                        
                    #info["initial"] = [line.to_JSON() for line in gh.lines if line != None]
                    #info["ghost_lines"] = [line.to_JSON() for line in predictions_to_emit]
                    #info["average_line"] = [p.to_JSON() for p in predictions]
                   
                    #info["comparison_line"] = [best_prediction.to_JSON()] if best_prediction is not None else []
                    #info["diffused_lines"] = [line.to_JSON() for idx, line in enumerate(gh.lines) if line is not None and idx in average_line.used_ids]
                    
                    #emit('prediction', info)
                    #print("prediction emitted")
                    #time.sleep(3)

                    gh.lines[line_idx] = None
                    

                    
                        
                            
                           

                                
                    
                        
                
                
            

def happyness_check(gh):

    for line in gh.lines:
        line.stopped = True
        line.is_fixed = True

    for idx in range(len(gh.lines)):
        if not gh.lines[idx].immutable:
            print("idx", idx)
            tmp = gh.lines[idx].clone()
            gh.lines[idx].stopped = False
            gh.lines[idx].is_fixed = False
            gh.calculate_gen_step(use_combinations=False, average_predictions=False, adaption_rate=1)
            gh.apply_gen_step()
            if gh.lines[idx] is None:
                print("line was None.")
                gh.lines[idx] = tmp
                continue

            print("pos_diff", gh.lines[idx].pos_diff(tmp))
            if gh.lines[idx].pos_diff(tmp) < 20:
                tmp.immutable = True

            gh.lines[idx] = tmp
            #else:
                #gh.lines[idx] = tmp



def toast(message):
    if message:
        print("toast", message)
        emit('toast', {'message': message})

@socketio.on('extend pattern')
def extend_pattern(data):
    
    #prediction durch alle linien laufen lassen, clustern und durchschnitt bilden oder in die richtung "wandern" lassen
    #für die ähnlihckeit distanz im vectorspace nehmen?
    #später dropout?
    
    
    print("LINES:", len(data["list"]))
    
    pt = PatternTrainer(data['name'])
    lineTrainer = LineTrainer(data['name'])
    info = {}
    
    if(len(data["list"]) <=0 ):
        gh.clear()
        
        #starten mit einem einzelnen set aus dem datensatz
        x = pt.getDatasetSample()
        
        base_list = []
        for i in range(x.size()[0]):
            n = GraphHandler.decompose_node_hidden_state(x[i])
            base_list.append( prediction2obj(n, lineTrainer) )
            

        #info["prediction"] = prediction2obj(pred, lineTrainer)
        #info["ground_truth"] = prediction2obj(ground_truth, lineTrainer)
        info["base_list"] = base_list

        emit('extention', info)
    else:
        #prediction durch alle linien laufen lassen, clustern und durchschnitt bilden oder in die richtung "wandern" lassen
        
        #bisherige linien in den graphhandler laden
        gh.add_raw(data)
        
        #latentspace auf den gleichen datensatz setzen
        gh.add_line_latentspace(lineTrainer)
        
        #alle linien abgehen, als referenz nehmen und prediction einsammeln
        samples = gh.sample_complete_graph(data['name'])
        predictions = []
        for sample in samples:
            x = sample["x"]
            edge_index = sample["edge_index"]
            ref_id = sample["ref_id"]
            z = pt.predict(x, edge_index)
            pred = GraphHandler.decompose_node_hidden_state(z)
            predictions.append( prediction2obj(pred, lineTrainer, ref_id) )
        
        
        info["prediction"] = predictions
        emit('extention', info)


#@socketio.on('train')
#def train(data):
#    trainer.trainModel(data['name'])

###### ROUTES

@app.route("/")
def start():
    return render_template('webcam.html')

@app.route("/latent-inspector")
def website_latent_inspector():
    return render_template('latent-inspector.html')

@app.route("/train")
def website_train():
    return render_template('train.html')

@app.route("/draw")
def website_draw():
    return render_template('draw.html')

@app.route("/webcam")
def website_webcam():
    return render_template('webcam.html')

@app.route("/save-svg", methods=["POST"])
def save_svg():
    payload = request.get_json(silent=True) or {}
    svg_content = payload.get("svg")

    if not isinstance(svg_content, str) or not svg_content.strip():
        return jsonify({"error": "Missing SVG content"}), 400

    # Keep file names predictable and filesystem-safe.
    model_name = str(payload.get("model", "pattern")).strip() or "pattern"
    safe_model_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name)
    timestamp = int(time.time() * 1000)
    filename = f"{safe_model_name}-{timestamp}.svg"

    output_dir = Path("./saved_svgs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(svg_content, encoding="utf-8")

    return jsonify({"status": "ok", "filename": filename}), 200

#@app.route("/photo")
#def website_photo():
#    return render_template('photo.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)


