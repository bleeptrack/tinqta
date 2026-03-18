'use strict';
import { SceneButton } from './SceneButton.js';
import { PaperCanvasDraw } from './PaperCanvasDraw.js';
import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js";

export class PatternTools extends HTMLElement {
	constructor(n) {
		
		super();
		this.socket = io();
		this.shadow = this.attachShadow({ mode: 'open' });
		this.canvas = new PaperCanvasDraw()
		this.basicTool = new paper.Tool()
		this.activeLine = null

		this.basicTool.onMouseDown = (event) => {
			this.activeLine = new paper.Path()
			this.activeLine.strokeColor = "black"
			this.activeLine.strokeWidth = 3
			this.activeLine.strokeCap = 'round'
			this.activeLine.add(event.point)
		}
		this.basicTool.onMouseDrag = (event) => {
			this.activeLine.add(event.point)
		}
		this.basicTool.onMouseUp = (event) => {
			this.activeLine.simplify()
			this.sendLine(this.activeLine)
			this.activeLine = null
		}
		
		this.socket.on("init", (config) => {
			console.log("config received", config)
			this.canvas.setConfig(config)
			
		})

		this.stampTool = new paper.Tool()
		this.stampTool.onMouseUp = (event) => {
			this.socket.emit("add:stamp", {"position": {"x": event.point.x, "y": event.point.y}, "name": this.shadow.getElementById("name").textContent, "correction": false})
		}

		this.stampCorrectionTool = new paper.Tool()
		this.stampCorrectionTool.onMouseUp = (event) => {
			this.socket.emit("add:stamp", {"position": {"x": event.point.x, "y": event.point.y}, "name": this.shadow.getElementById("name").textContent, "correction": true})
		}

		this.visualTool = new paper.Tool()
		this.visualTool.onMouseDown = (event) => {
			this.activeLine = new paper.Path()
			this.activeLine.strokeColor = "black"
			this.activeLine.strokeWidth = 3
			this.activeLine.strokeCap = 'round'
			this.activeLine.add(event.point)
		}
		this.visualTool.onMouseDrag = (event) => {
			this.activeLine.add(event.point)
		}
		this.visualTool.onMouseUp = (event) => {
			this.activeLine.simplify()
			let processedLine = this.canvas.processLine(this.activeLine)
			this.socket.emit("add:visual", {"line": processedLine, "name": this.shadow.getElementById("name").textContent, "correction": false})
			this.activeLine = null
		}

		this.visualCorrectionTool = new paper.Tool()
		this.visualCorrectionTool.onMouseDown = (event) => {
			this.activeLine = new paper.Path()
			this.activeLine.strokeColor = "black"
			this.activeLine.strokeWidth = 3
			this.activeLine.strokeCap = 'round'
			this.activeLine.add(event.point)
		}
		this.visualCorrectionTool.onMouseDrag = (event) => {
			this.activeLine.add(event.point)
		}
		this.visualCorrectionTool.onMouseUp = (event) => {
			this.activeLine.simplify()
			let processedLine = this.canvas.processLine(this.activeLine)
			this.socket.emit("add:visual", {"line": processedLine, "name": this.shadow.getElementById("name").textContent, "correction": true})
			this.activeLine = null
		}
		
		
		this.socket.on("init", (config) => {
			console.log("config received", config)
			this.canvas.setConfig(config)
			
		})

		this.socket.on('toast', (data) => {
			console.log(`MESSAGE: ${data.message}`)
			
		})

		this.socket.on('draw:lines', (data) => {
			this.canvas.clear()
			console.log("draw:lines received", data)
			for(let line of data.lines){
				this.canvas.drawLine(line, "red")
			}
		})


		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			<link rel="stylesheet" href="/static/style.css">
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				#container{
					display: flex;
					flex-direction: column;
					height: 100%;
					box-sizing: border-box;
					padding: 5%;
					gap: 1vh;
				}
				#canvas-container{
					width: 100%;
					flex: 1;
					border: 2px solid black;
					position: relative;
					min-height: 0;
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
					flex-direction: row;
					gap: 1vh;
				}
			</style>
			
			<div id="container">
				<h1>Draw!</h1>
				<div id="name" class="scribble input" placeholder="Enter your model name" contenteditable=true></div>
				<button id="stamp">stamp</button>
				<button id="stampCorrection">stamp correction</button>
				<button id="visual">visual</button>
				<button id="visualCorrection">visual correction</button>
				<button id="draw">draw</button>
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
		this.shadow.getElementById("stamp").addEventListener("click", () => {
			this.stampTool.activate()
		})
		this.shadow.getElementById("stampCorrection").addEventListener("click", () => {
			this.stampCorrectionTool.activate()
		})
		this.shadow.getElementById("draw").addEventListener("click", () => {
			this.basicTool.activate()
		})
		this.shadow.getElementById("visual").addEventListener("click", () => {
			this.visualTool.activate()
		})
		this.shadow.getElementById("visualCorrection").addEventListener("click", () => {
			this.visualCorrectionTool.activate()
		})
	}


	sendLine(line) {
		let processedLine = this.canvas.processLine(line)
		this.socket.emit("add:line", processedLine)
		console.log("sent line", processedLine)
	}
	
}

customElements.define('pattern-tools', PatternTools);
