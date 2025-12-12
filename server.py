from flask import Flask, render_template
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



#path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'dataset-test')
#dataset = MyOwnDataset("testdata", path)






@socketio.event
def connect():
    print("User connected")
    emit('init', config)
    mlist = getModels()
    print(mlist)
    emit('models', mlist)

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

    validation_dataset = gh.calculate_dataset_onthefly(nr_samples=len(base_dataset)*0.2)
    pt.setup_from_test_sample(validation_dataset)
    
    threshold = 5
    count = 0
    
    
    data_jitter = 0
    dataset = base_dataset + noisy_dataset

    for i in range(1001):
        if count >= threshold:
            noisy_dataset = gh.calculate_dataset_onthefly(nr_samples=len(base_dataset))
            dataset = base_dataset + noisy_dataset
            threshold -= 1
            count = 0
            if threshold <= 3:
                threshold = 3

        if i > 25:
            data_jitter = min(0.02, (i - 25) * 0.02 / 50)  # Ramp over 75 epochs
        pt.trainModel(dataset, data_jitter=data_jitter)
        count += 1

         # Check if learning rate has reached minimum
        current_lr = pt.optimizer.param_groups[0]['lr']
        min_lr = pt.scheduler.min_lr if pt.scheduler is not None else 0
        if current_lr <= min_lr:
            print(f"Stopping training: Learning rate reached minimum ({current_lr})")
            break

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
                all_line_lists.append(lines)

        # Iterate backwards to avoid skipping elements when removing items
        for i in range(len(gh.lines) - 1, -1, -1):
            line = gh.lines[i]
            if not line.immutable:
                gh.lines.pop(i)

        line_deposit = all_line_lists.pop(0)
        
        

        
        
        print("line_deposit", len(line_deposit))
        

    else:

        line_changed = False
        change_count = 6
        loopcount = 0
        for j in range(1500):

            if loopcount > 15 or (len(all_line_lists) == 0 and loopcount > 5):
                # Remove all lines from gh.lines that are not immutable
                #all_line_lists.append([line for line in gh.lines if not getattr(line, "immutable", False)])
                gh.lines = [line for line in gh.lines if getattr(line, "immutable", False)]
                loopcount = 0
                

            
            if ( all(line.immutable for line in gh.lines) or change_count > 5):
                if len(all_line_lists) > 0:
                    # Keep the first one, remove the rest
                    change_count = 0
                    loopcount = 0
                    gh.lines.extend(all_line_lists.pop(0))
                #else:
                #    gh.start_new_line()

            loopcount += 1
                
            
            
           
            indices = [i for i, line in enumerate(gh.lines) if not getattr(line, "immutable", False)]
            random.shuffle(indices)
            for diff_idx in indices:
                info = {}
                # Initialize info with current state to prevent empty emits
                info["initial"] = [line.to_JSON() for line in gh.lines]
                info["ghost_lines"] = []

                # Randomly select 10% of all lines
                gh.ghost_lines = []
                # Initialize stop_count before creating backup so it's preserved
                for line in gh.lines:
                    if not hasattr(line, "stop_count"):
                        line.stop_count = 0
                backup_lines = [line.clone() for line in gh.lines]

                #num_lines_to_select = max(1, int(len(gh.lines) * 0.01))
                #selected_lines = random.sample(gh.lines, min(num_lines_to_select, len(gh.lines)))
                
                #ToDo: vermutlich verliere ich ne linie wenn sie rauslauft und dann stimmt der index nicht mehr
                #die gleiche linie wird auch auf jeden fall mehrfach ausgewählt
                print("gh.lines", len(gh.lines))
                print("diffusing line", diff_idx)
                gh.lines[diff_idx].stopped = False
                gh.lines[diff_idx].is_fixed = False

                
                
                #emit('prediction', info)

                count = 0

                
                
                while gh.calculate_gen_step(use_combinations=False, average_predictions=False):
                    info["initial"] = [line.to_JSON() for line in gh.lines]
                    info["ghost_lines"] = [line.to_JSON() for line in gh.ghost_lines]
                    
                    count += 1
                    
                    if count > 50:
                        toast("LOOP LIMIT reached")
                        gh.gen_step = []
                        break
                    
                    if count % 20 == 0:
                        emit('prediction', info)
                        
                        
                    gh.apply_gen_step()

                
                # Update ghost_lines in case loop didn't execute
                info["ghost_lines"] = [line.to_JSON() for line in gh.ghost_lines]
                emit('prediction', info)

                for idx, line in enumerate(gh.lines):
                    if line is None:
                        print("line was None. restoring from backup", idx)
                        gh.lines[idx] = backup_lines[idx]
                        continue
                    if not line.is_fixed:
                        if not line.stopped:
                            if idx < len(backup_lines):
                                print("line not fixed or stopped. restoring from backup", idx)
                                
                                # Preserve stop_count when restoring from backup
                                #if hasattr(line, "stop_count"):
                                #    backup_lines[idx].stop_count = line.stop_count
                                #gh.lines[idx] = backup_lines[idx]
                                line.stopped = True
                                line.is_fixed = True
                            else:
                                print("line not found in backup. removing", idx)
                                gh.lines.remove(line)
                        else:
                            diff = line.pos_diff(backup_lines[idx])
                            if not hasattr(line, "stop_count"):
                                line.stop_count = 0
                            
                            print("line is stopped.", diff)
                            if diff < 10:
                                line.stop_count += 1
                                if line.stop_count > 2:
                                    print("line is close enough. making immutable", diff)
                                    line.immutable = True
                                


                # Compare each line with its corresponding backup (matching indices only)
                min_len = min(len(gh.lines), len(backup_lines))
                accumulated_diff = [gh.lines[i].pos_diff(backup_lines[i]) for i in range(min_len)]
                
                if sum(accumulated_diff) < 0.1:
                    print("lines are the same. no change.", sum(accumulated_diff))
                    change_count += 1
                    line_changed = False
                else:
                    change_count = 0
                    line_changed = True
               
                
                
                    
                for line in gh.lines:
                    line.stopped = True
                    line.is_fixed = True

                for line in gh.ghost_lines:
                    line.stopped = True
                    line.is_fixed = True

                gh.lines.extend(gh.ghost_lines)
                
                info["initial"] = [line.to_JSON() for line in gh.lines]
                emit('prediction', info)
            

            


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

@app.route("/webcam")
def website_webcam():
    return render_template('webcam.html')

#@app.route("/photo")
#def website_photo():
#    return render_template('photo.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)


