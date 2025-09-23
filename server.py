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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

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



#path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'dataset-test')
#dataset = MyOwnDataset("testdata", path)






@socketio.event
def connect():
    print("User connected")
    emit('init', config)
    mlist = getModels()
    print(mlist)
    emit('models', mlist)

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



@socketio.on('train pattern')
def new_pattern(data):
    print("hallo", data)
    lineTrainer = LineTrainer(data['name'])
    
    gh.clear()
    gh.init_lines(data['list'])
    gh.set_default_trainers(line_trainer=lineTrainer)
        
    gh.add_line_latentspace()
   
    gh.save_pattern_training_data(data['name'])

    pt = PatternTrainer(data['name'])
    pt.trainModel(send_progress_pattern)

def send_progress_pattern(name):
    
    data = {'name': name}
    print("sending progress pattern", data)
    sample_pattern(data)


# @socketio.on('sample pattern')
# def sample_pattern(data):
#     pt = PatternTrainer(data['name'])
#     lineTrainer = LineTrainer(data['name'])

#     #bisherige linien in den graphhandler laden
#     gh.add_raw(data)
        
#     #latentspace auf den gleichen datensatz setzen
#     gh.add_line_latentspace(lineTrainer)
        
#     #alle linien abgehen, als referenz nehmen und prediction einsammeln
#     samples = gh.sample_complete_graph(data['name'])
#     predictions = []
#     info = {}
#     for sample in samples:
#         x = sample["x"]
#         edge_index = sample["edge_index"]
#         ref_id = sample["ref_id"]
#         z = pt.predict(x, edge_index)
#         pred = GraphHandler.decompose_node_hidden_state(z)
#         predictions.append( prediction2obj(pred, lineTrainer, ref_id) )
        
        
#     info["prediction"] = predictions
#     emit('extention', info)



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

    z,y,x = pt.generate()
    pred = gh.decompose_node(z)
    ground_truth = gh.decompose_node(y)
    base_list = []
    for i in range(x.size()[0]):
        n = gh.decompose_node(x[i])
        base_list.append(n.to_JSON())

    info = {}
    info["prediction"] = pred.to_JSON()
    info["ground_truth"] = ground_truth.to_JSON()
    info["base_list"] = base_list

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

    if(len(gh.lines) == 0):
        lineTrainer = LineTrainer(data['name'])
        pt = PatternTrainer(data['name'])
        gh.clear()
        gh.set_default_trainers(pattern_trainer=pt, line_trainer=lineTrainer)
        
        gh.calculate_original_lines()
        gh.calculate_line_thresholds()
        gh.init_original()

        #gh.random_fill()

        info = {}
        info["base_list"] = [line.to_JSON() for line in gh.lines]

        emit('prediction', info)

    else:

        
        info = {}
        info["initial"] = [line.to_JSON() for line in gh.lines]
        
        

        gh.reject_abnormal_lines()
        gh.start_new_line()
        
        
       
        
        count = 0
        while gh.calculate_gen_step():
            info["initial"] = [line.to_JSON() for line in gh.lines]
            info["ghost_lines"] = [line.to_JSON() for line in gh.ghost_lines]
            count += 1
            if count % 10 == 0:
                print("loop count", count)
            if count > 50:
                toast("LOOP LIMIT reached")
                gh.gen_step = []
                break
            
            emit('prediction', info)
                

            
            gh.apply_gen_step()
            socketio.sleep(0.01) 

        gh.choose_ghost_lines()
        info["top_p"] = [line.to_JSON() for line in gh.ghost_lines]
        print("SERVER: after top_p", len(gh.ghost_lines), len(info["top_p"]))
        emit('prediction', info)
        untouched_lines, not_matched, merged_lines = gh.combine_ghost_and_main_lines()

        not_matched.sort(key=lambda x: x.averaged_from)
        print([line.averaged_from for line in not_matched])

        #die top auswahl müsste am ende eigentlich auf die nicht schon vorhandenen linien angewendet werden?
        not_matched = gh.top_p(not_matched, 0.7)
        print([line.averaged_from for line in not_matched])

        
        gh.lines = not_matched  + merged_lines + untouched_lines

        gh.reject_abnormal_lines()
        info["untouched_lines"] = [line.to_JSON() for line in untouched_lines]
        info["not_matched"] = [line.to_JSON() for line in not_matched]
        info["merged_lines"] = [line.to_JSON() for line in merged_lines]
        emit('prediction', info)

        for line in gh.lines:
            line.stopped = True
            line.is_fixed = True

        # for i in range(200):
        #     gh.self_arrange()
        #     info["diffused_lines"] = [line.to_JSON() for line in gh.lines]
        #     emit('prediction', info)

        # gh.reject_abnormal_lines()

        info["diffused_lines"] = [line.to_JSON() for line in gh.lines]
        emit('prediction', info)
        

        # for i in range(300):
        #     print("DIFFUSING ROUND", i)
            
        #     gh.self_arrange()
        #     info["diffused_lines"] = [line.to_JSON() for line in gh.lines]
        #     emit('prediction', info)
            

            


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


