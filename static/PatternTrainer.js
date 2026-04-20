'use strict';
import { SceneButton } from './SceneButton.js';
import { PaperCanvas } from './PaperCanvas.js';
import { ProgressBar } from './ProgressBar.js';
import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js";

export class PatternTrainer extends HTMLElement {
	constructor(n) {
		
		super();
		this.socket = io();
		this.shadow = this.attachShadow({ mode: 'open' });
		this.canvas = new PaperCanvas()
		this.epochLines = []
		
		
		this.socket.on("init", (config) => {
			console.log("config received", config)
			this.canvas.setConfig(config)
			this.canvas.setDrawingGuides()
			
		})
		
		this.socket.on("progress", (text) => {
			if(this.canvas.placeholder){
				this.canvas.placeholder.remove()
			}
			console.log("progress received", text)
			if(text.lines){				
				this.canvas.trainingEpoch(text)
				this.epochLines.push(text)
				
			}else if(text.percent){
				this.progressbar.setPercentage(Number(text.percent), text.label)
			}
		})

		this.socket.on('toast', (data) => {
			console.log(`MESSAGE: ${data.message}`)
			
		})

		this.socket.on('result', (data) => {
			console.log(data)
			for(let idx in data.list){
				console.log(data.list[idx])
				let drawnLine = this.canvas.drawLine(data.list[idx], "black")
				drawnLine.position = paper.view.center
				//drawnLine.scale(data.scales[idx])
				//drawnLine.rotate(data.rotations[idx]*360)
			}
		});

	this.socket.on('prediction', (data) => {
		if(paper.project.layers["sample_nodes"]){
			paper.project.layers["sample_nodes"].remove()
		}
		if(paper.project.layers["ground_truth"]){
			paper.project.layers["ground_truth"].remove()
		}
		if(paper.project.layers["prediction"]){
			paper.project.layers["prediction"].remove()
		}
		
		this.canvas.clear()
		let baseLines = []
		if(data["patch_data"]){
			let x = data["patch_data"]["x"]
			let y = data["patch_data"]["y"]
			let patch_distance = data["patch_data"]["distance"]
			let outsider_distance = data["patch_data"]["outsider_distance"]
			for(let i = 0; i < x; i++){
				for(let j = 0; j < y; j++){
					let c = new Path.Rectangle(0, 0, outsider_distance*2, outsider_distance*2)
					c.position = new Point(i * patch_distance, j * patch_distance)
					c.fillColor = 'grey'
					c.opacity = 0.5
				}
			}
		}
		if(data["initial"]){
			for(let line of data["initial"]){
				let color = "black"
				if(line["is_fixed"]){
					color = "orange"
				}
				if(line["immutable"]){
					color = "red"
				}
				if(line["added_at_stage"]){
					if(line["added_at_stage"] == "patch accept"){
						color = "pink"
					}
					if(line["added_at_stage"] == "weave"){
						color = "blue"
					}
					if(line["added_at_stage"] == "weave adjust"){
						color = "purple"
					}
					if(line["added_at_stage"] == "patch"){
						color = "black"
					}
				}
				let l = this.canvas.drawLine(line, color, paper.project.layers["lines"])
				l.strokeWidth = 7
				l.opacity = 0.2
				l.strokeCap = "round"
				baseLines.push(l)
			}
		}
		if(data["base_list"]){
			
			for(let line of data["base_list"]){
				if(line["is_fixed"]){
					let l = this.canvas.drawLine(line, "orange", paper.project.layers["lines"])
					l.strokeWidth = 15
					l.opacity = 0.5
					baseLines.push(l)
				}
				//l.translate(paper.view.center)
				
			}
		}
		// Old key for backward compatibility
		if(data["pos"]){
			let c = new Path.Circle(data["pos"], 20)
			c.fillColor = "red"
			//l.translate(paper.view.center)
		}
		
		// Sample visualization - new proper naming
		if(data["sample_nodes"]){
			let layer = new paper.Layer({name: "sample_nodes"})
			for(let line of data["sample_nodes"]){
				let l = this.canvas.drawLine(line, "blue", false, layer)
				l.opacity = 0.3
				l.strokeWidth = 20
				l.strokeCap = "round"
				
			}
			
		}
		if(data["dropped_out_nodes"]){
			/*
			for(let pos of data["dropped_out_nodes"]){
				// Draw a small circle at the dropped-out position
				let c = new Path.Circle(pos, 15)
				c.fillColor = "cyan"
				c.opacity = 0.2
				c.strokeColor = "cyan"
				c.strokeWidth = 2
			}
			*/
		}
		if(data["target_pos"]){
			/*
			for(let pos of data["target_pos"]){
				let c = new Path.Circle(pos, 15)
				c.fillColor = "green"
				c.opacity = 0.5
			}
				*/
		}
		if(data["ground_truth"]){
			let layer = new paper.Layer({name: "ground_truth"})
			for(let line of data["ground_truth"]){
				let l = this.canvas.drawLine(line, "yellow", false, layer)
				l.opacity = 0.5
				l.strokeWidth = 20
				l.strokeCap = "round"
			}
		}
		if(data["flow_data"]){
			for(let pos of data["flow_data"]){
				let c2 = new Path.Circle(pos["pred_x"], pos["pred_y"], 3)
				c2.fillColor = "purple"
				c2.opacity = 0.5


				let vec = new Point(pos["pred_x"], pos["pred_y"]).subtract(new Point(pos["x"], pos["y"]))
				if(vec.length > 10){
					vec = vec.normalize().multiply(10)
				}
				let l = new Path.Line(pos["x"], pos["y"], pos["x"] + vec.x, pos["y"] + vec.y)
				l.strokeColor = "blue"
				l.opacity = 0.7
				let c = new Path.Circle(pos["x"], pos["y"], 2)
				c.fillColor = "blue"
				console.log("drawing flow field at", pos["x"], pos["y"])


				let line = this.canvas.drawLine(pos["line"], "black")
				
				
			}
		}
		// Other visualizations used by generate pattern
		if(data["ghost_lines"]){
			let clusterColors = ["red", "blue", "purple", "orange", "pink", "brown", "grey", "black"]
			for(let line of data["ghost_lines"]){
				let l = this.canvas.drawLine(line, clusterColors[line["cluster_number"]])
				l.opacity = 0.3
				l.strokeWidth = 10
				l.strokeCap = "round"

				//l.translate(paper.view.center)
			}
		}
		if(data["top_p"]){
			for(let line of data["top_p"]){
				let l = this.canvas.drawLine(line, "green")
				l.opacity = 0.3
				l.strokeWidth = 10
				l.strokeCap = "round"
				//l.translate(paper.view.center)
			}
		}
		
		// Old keys for backward compatibility (deprecated)
		if(data["untouched_lines"]){
			let layer = new paper.Layer({name: "untouched_lines"})
			for(let line of data["untouched_lines"]){
				let l = this.canvas.drawLine(line, "red")
				l.opacity = 0.3
				l.strokeWidth = 20
				l.strokeCap = "round"
				layer.add(l)
			}
			paper.project.addLayer(layer)
		}
		if(data["not_matched"]){
			for(let line of data["not_matched"]){
				let l = this.canvas.drawLine(line, "yellow")
				l.opacity = 0.3
				l.strokeWidth = 20
				l.strokeCap = "round"
			}
		}
		
		if(data["comparison_line"]){
			for(let line of data["comparison_line"]){
				let l = this.canvas.drawLine(line, "blue")
				l.opacity = 0.5
				l.strokeWidth = 7
				l.strokeCap = "round"
			}
		}
		if(data["average_line"]){
			for(let line of data["average_line"]){
				let l = this.canvas.drawLine(line, "green")
				l.opacity = 0.4
				l.strokeWidth = 7
				l.strokeCap = "round"
			}
		}
		if(data["diffused_lines"]){
			for(let line of data["diffused_lines"]){
				let l = this.canvas.drawLine(line, "black")
				l.opacity = 1
				l.strokeWidth = 7
				l.strokeCap = "round"
			}
		}
		if(data["prediction"]){
			let layer = new paper.Layer({name: "prediction"})
			if(Array.isArray(data["prediction"])){
				for(let line of data["prediction"]){
					let l = this.canvas.drawLine(line, "red", false, layer)
					l.opacity = 0.5
					l.strokeWidth = 20
					l.strokeCap = "round"
					/*
					if(line["used_ids"] && !line["is_fixed"]){
						
						for(let id of line["used_ids"]){
							baseLines[id].strokeColor = "green"
							baseLines[id].strokeWidth = 10
							baseLines[id].opacity = 0.5
							baseLines[id].strokeCap = "round"
						}
						
						let c = new Path.Circle(l.position, this.canvas.config["max_dist"])
						c.fillColor = "grey"
						c.opacity = 0.5
						c.sendToBack()
						this.canvas.centerDrawing(c.position)
						
					}
					*/
					//l.translate(paper.view.center)
				}
			}else{
				let l = this.canvas.drawLine(data["prediction"], "red", false, layer)
				l.opacity = 0.5
				l.strokeWidth = 20
				l.strokeCap = "round"
				//l.translate(paper.view.center)
			}
		}
		
		this.canvas.centerDrawing()

		paper.project.layers["background"]?.remove()
		paper.project.layers["guides"]?.remove()

		//this.canvas.downloadSVG()
		//this.drawArt(data)
	});
		
		

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			<link rel="stylesheet" href="/static/style.css">
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				#container{
					display: flex;
					flex-direction: row;
					height: 100%;
					width: 100%;
					box-sizing: border-box;
					padding: 5%;
					gap: 1vh;
				}
				#canvas-container{
					width: 60vw;
					height: 100%;
					flex: 1;
					border: 2px solid black;
					position: relative;
					min-height: 0;

				}
				#canvas-container canvas{
					width: 100%;
					height: 100%;
				}
				input{
					max-width: 40vw;
					width: 50em;
					height: 2em;
					padding: 0.5em;
					align-self: center;
				}
				#undo{
					position: absolute;
					top: 2vh;
					right: 2vh;
					width: 2em;
				}
				#train{
					align-self: center;
				}
				[contenteditable=true]:empty:before {
					content: attr(placeholder);
					pointer-events: none;
					display: block; /* For Firefox */
					color: grey;
				}
				#button-container{
					display: flex;
					flex-direction: column;
					gap: 1vh;
					width: 100%;
				}
			</style>
			
			<div id="container">
				
				<aside id="aside">
					<h1>Train your Scribble Model</h1>
					<div id="name" class="scribble input" placeholder="Enter your model name" contenteditable=true></div>
					
					<div id="button-container">
						<button id="train" class="scribble">train</button>
						<button id="test-lines" class="scribble">test lines</button>
						<button id="pattern-sample" class="scribble">get pattern sample</button>
						<button id="train-pattern" class="scribble">train pattern</button>
						<button id="generate-pattern" class="scribble">generate pattern</button>
						<button id="make-noise" class="scribble">add noise</button>
						<button id="reset-zoom" class="scribble">reset zoom</button>
					</div>
				</aside>
				<div id="canvas-container">
					<button id="undo" class="material-symbols-outlined scribble">undo</button>
				</div>
			</div>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));

		this.shadow.getElementById("canvas-container").appendChild(this.canvas)
		this.shadow.getElementById("undo").addEventListener("click", () => {
			this.canvas.undo()
		})
		
		this.shadow.getElementById("train").addEventListener("click", () => {
			console.log(this.canvas.linelist)
			console.log("name:", this.shadow.getElementById("name").innerHTML)
			
			this.socket.emit("new dataset", {
				name: this.shadow.getElementById("name").innerHTML,
				list: this.canvas.linelist,
			})
			
			this.progressbar = new ProgressBar()
			this.shadow.getElementById("train").replaceWith(this.progressbar)
			
		})

		this.shadow.getElementById("train-pattern").addEventListener("click", () => {
			this.socket.emit("train pattern", {
				name: this.shadow.getElementById("name").innerHTML,
				list: this.canvas.linelist,
			})
		})

		this.shadow.getElementById("test-lines").addEventListener("click", () => {
			this.socket.emit('generate', {nr: 200, name:this.shadow.getElementById("name").innerHTML})
		})

		this.shadow.getElementById("pattern-sample").addEventListener("click", () => {
			this.socket.emit('sample pattern', {name:this.shadow.getElementById("name").innerHTML})
		})

		this.shadow.getElementById("generate-pattern").addEventListener("click", () => {
			this.socket.emit('generate pattern', {name:this.shadow.getElementById("name").innerHTML})
		})

		this.shadow.getElementById("make-noise").addEventListener("click", () => {
			this.socket.emit("make noise", {noise_level: 0.02})
		})

		this.shadow.getElementById("reset-zoom").addEventListener("click", () => {
			this.canvas.resetZoom();
		})
		
	}

	drawArt(data){
		paper.project.layers["lines"].removeChildren()

		if(data["prediction"]){
			if(Array.isArray(data["prediction"])){
				for(let line of data["prediction"]){
					if(!line["is_fixed"]){
						let l = this.canvas.drawLine(line, "blue", true, paper.project.layers["art"])
						l.strokeWidth = 10
						l.opacity = 0.005
					}
				}
			}
			
		}
	}

	connectedCallback() {
		window.downloadLines = this.saveLines.bind(this)
		//this.canvas.setPlaceholder()
	}
	
	saveLines(){
		let link = document.createElement('a');
		link.download = 'tinqta-training.json';
		let data = {
			"training": this.epochLines,
			"linelist": this.canvas.linelist,
			"original": this.canvas.originalLines
		}
		link.href = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data));
		link.click();
	}

}

customElements.define('pattern-trainer', PatternTrainer);
