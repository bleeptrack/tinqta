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
        gh.init_original()
        

        info = {}
        info["base_list"] = [line.to_JSON() for line in gh.lines]

        emit('prediction', info)

    else:
        for i in range(1):
            gh.calculate_gen_step()
            info = {}
            info["base_list"] = [line.to_JSON() for line in gh.lines]
            info["prediction"] = [line.to_JSON() for line in gh.gen_step]

            emit('prediction', info)

            gh.lines = []
            #gh.apply_gen_step()
            #socketio.sleep(0.05)  # 50ms delay

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


