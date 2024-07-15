from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
from DrawData import GraphHandler
from Model import LineTrainer, PatternTrainer
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

gh = GraphHandler()



#path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'dataset-test')
#dataset = MyOwnDataset("testdata", path)


def tensor2Points(x):
    points = []

    for i in range(x.shape[0]):
        points.append({
            'x': x[i][0].item(),
            'y': x[i][1].item()
            })
    return points

def prediction2obj(pred, lineTrainer, ref_id=None):
    line_obj = {}
    tensor = lineTrainer.decode_latent_vector(pred["latVec"])
    line_obj['points'] = tensor2Points(tensor)
    line_obj['scale'] = pred["scale"]
    line_obj['rotation'] = pred["rot"]
    line_obj['position'] = {
        "x": pred['posX'],
        "y": pred['posY']
        }
    if ref_id:
        line_obj['reference_id'] = ref_id
    return line_obj

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
        gh.init_raw(data)
        gh.save_line_training_data(data['name'])

        lineTrainer = LineTrainer(data['name'])
        lineTrainer.trainModel(send_progress)
        
def send_progress(trainer, text, label=None):
   
    if isinstance(text, list):
        pointlist = []
        for z in text:
            print(z)
            tensor = trainer.decode_latent_vector(z)
            pointlist.append(tensor2Points(tensor))
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
    print("RAW", gh.raw_data)
    
    if data['name'] == "random":
        print("choosing RANDOM model")
        trainer = LineTrainer(random.choice(getModels()))
    else:
        trainer = LineTrainer(data['name'])

    tensors, scales, rotations = trainer.generate(data['nr'], 0.5)
    pointlist = []

    for tensor in tensors:
        pointlist.append(tensor2Points(tensor))

    emit('result', {'list': pointlist, 'scales':scales, 'rotations':rotations})
    
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
        pointlist.append(tensor2Points(latentLine))
    
    

    #for tensor in tensors:
    #    pointlist.append(tensor2Points(tensor))

    emit('latentLine', {'list': pointlist})
    
@socketio.on('compare')
def compare(data):
    trainer = LineTrainer(data['name'])
    
    pointlist = []
    originlist = []
    tensors, originpoints = trainer.extractOriginLineVectors()
    for z in tensors:
        print(z)
        tensor = trainer.decode_latent_vector(z)
        pointlist.append(tensor2Points(tensor))
        
    for ori in originpoints:
        originlist.append(tensor2Points(ori))
        
    emit('result', {'list': pointlist, "origins":originlist})



@socketio.on('train pattern')
def new_pattern(data):
    print("hallo", data)
    lineTrainer = LineTrainer(data['name'])
    gh.add_line_latentspace(lineTrainer)
    gh.save_pattern_training_data(data['name'], data['name'])

    #todo training hier ist broken ||||| ist das so?
    pt = PatternTrainer(data['name'])
    pt.trainModel()


@socketio.on('generate pattern')
def generate_pattern(data):

    pt = PatternTrainer(data['name'])
    lineTrainer = LineTrainer(data['name'])

    z,y,x = pt.generate()
    pred = GraphHandler.decompose_node_hidden_state(z)
    ground_truth = GraphHandler.decompose_node_hidden_state(y)
    base_list = []
    for i in range(x.size()[0]):
        n = GraphHandler.decompose_node_hidden_state(x[i])
        base_list.append( prediction2obj(n, lineTrainer) )

    info = {}
    info["prediction"] = prediction2obj(pred, lineTrainer)
    info["ground_truth"] = prediction2obj(ground_truth, lineTrainer)
    info["base_list"] = base_list

    emit('prediction', info)

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


@socketio.on('train')
def train(data):
    trainer.trainModel(data['name'])

###### ROUTES

@app.route("/")
def start():
    return render_template('index.html')

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


