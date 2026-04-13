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
		this.activeModelName = 'grid'
		this.activeCorrection = false
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
			this.socket.emit("add:stamp", {"position": {"x": event.point.x, "y": event.point.y}})
		}

		this.stampCorrectionTool = new paper.Tool()
		this.stampCorrectionTool.onMouseUp = (event) => {
			this.socket.emit("add:stamp", {"position": {"x": event.point.x, "y": event.point.y}})
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
			this.socket.emit("add:visual", {"position": {"x": event.point.x, "y": event.point.y},"line": processedLine, "name": this.activeModelName, "correction": false})
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
			this.socket.emit("add:visual", {"position": {"x": event.point.x, "y": event.point.y},"line": processedLine, "name": this.activeModelName, "correction": true})
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
				:host {
					display: block;
					height: 100%;
					box-sizing: border-box;
				}
				#container{
					display: flex;
					flex-direction: row;
					align-items: stretch;
					height: 100%;
					box-sizing: border-box;
					padding: 2vh 2vw;
					gap: 2vw;
					min-height: 0;
				}
				#canvas-area {
					flex: 1;
					min-width: 0;
					display: flex;
					flex-direction: column;
					min-height: 0;
				}
				#canvas-container{
					width: 100%;
					flex: 1;
					border: 2px solid black;
					position: relative;
					min-height: 0;
				}
				input{
					max-width: 100%;
					width: 100%;
					height: 2em;
					padding: 0.5em;
					box-sizing: border-box;
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
				#sidebar {
					flex: 0 0 25%;
					max-width: 25%;
					min-width: 0;
					display: flex;
					flex-direction: column;
					gap: 1.25rem;
					overflow-x: hidden;
					overflow-y: auto;
					min-height: 0;
					padding-inline: 0.35rem;
					box-sizing: border-box;
				}
				.button-section {
					display: flex;
					flex-direction: column;
					gap: 0.5rem;
					min-width: 0;
				}
				.button-section h2 {
					margin: 0;
					font-size: 0.85rem;
					font-weight: 700;
					text-transform: uppercase;
					letter-spacing: 0.04em;
					color: #333;
					font-family: ui-sans-serif, system-ui, sans-serif;
				}
				#tools-container,
				#model-container {
					display: flex;
					flex-direction: column;
					gap: 0.5rem;
					min-width: 0;
				}
				label.toggle-row.scribble {
					width: 100%;
					max-width: 100%;
					min-width: 0;
					box-sizing: border-box;
					margin: 0;
					cursor: pointer;
					display: flex;
					align-items: center;
					justify-content: center;
					text-align: center;
					overflow-wrap: break-word;
					white-space: normal;
					hyphens: auto;
				}
				label.toggle-row.scribble input {
					position: absolute;
					opacity: 0;
					width: 1px;
					height: 1px;
					margin: -1px;
					overflow: hidden;
					clip: rect(0, 0, 0, 0);
					white-space: nowrap;
					border: 0;
					pointer-events: none;
				}
				label.toggle-row.scribble .toggle-row-label {
					pointer-events: none;
				}
				label.toggle-row.scribble:has(input:checked) {
					filter: brightness(0.88);
					box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.18);
				}
				label.toggle-row.scribble:has(input:focus-visible) {
					outline: 2px solid #000;
					outline-offset: 2px;
				}
				label.toggle-row.scribble:has(input:checked)::after {
					bottom: 2px;
					left: 2px;
					width: calc(100% - 4px);
				}
				#tools-container label.scribble::after,
				#model-container label.scribble::after {
					width: calc(100% - 8px);
					left: 4px;
					box-sizing: border-box;
				}
				#tools-container label.scribble:hover::after,
				#model-container label.scribble:hover::after {
					width: calc(100% - 4px);
					left: 2px;
				}
				#tools-container label.scribble:has(input:checked):hover::after,
				#model-container label.scribble:has(input:checked):hover::after {
					width: calc(100% - 4px);
					left: 2px;
				}
			</style>
			
			<div id="container">

				<!-- <div id="name" class="scribble input" placeholder="Enter your model name" contenteditable=true></div> -->
				
				<div id="canvas-area">
					<div id="canvas-container">
						<button id="undo" class="material-symbols-outlined scribble">undo</button>
					</div>
				</div>
				<aside id="sidebar">
					<section class="button-section" aria-label="Tools">
						<h2 id="tools-heading">Tools</h2>
						<div id="tools-container" role="radiogroup" aria-labelledby="tools-heading">
							<label class="scribble toggle-row"><input type="radio" name="pattern-tool" value="stamp"><span class="toggle-row-label">stamp</span></label>
							<label class="scribble toggle-row"><input type="radio" name="pattern-tool" value="visual"><span class="toggle-row-label">visual</span></label>
							<label class="scribble toggle-row"><input type="radio" name="pattern-tool" value="draw" checked><span class="toggle-row-label">draw</span></label>
						</div>
					</section>
					<section class="button-section" aria-label="Models">
						<h2 id="models-heading">Models</h2>
						<div id="model-container" role="radiogroup" aria-labelledby="models-heading">
							<label class="scribble toggle-row"><input type="radio" name="pattern-model" value="boxes"><span class="toggle-row-label">boxes</span></label>
							<label class="scribble toggle-row"><input type="radio" name="pattern-model" value="swirls"><span class="toggle-row-label">swirls</span></label>
							<label class="scribble toggle-row"><input type="radio" name="pattern-model" value="grid" checked><span class="toggle-row-label">grid</span></label>
							<label class="scribble toggle-row"><input type="radio" name="pattern-model" value="fence"><span class="toggle-row-label">fence</span></label>
						</div>
					</section>
				</aside>
			</div>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));

		this.shadow.getElementById("canvas-container").appendChild(this.canvas)
		this.shadow.getElementById("undo").addEventListener("click", () => {
			this.canvas.undo()
		})

		const toolByValue = {
			stamp: this.stampTool,
			stampCorrection: this.stampCorrectionTool,
			visual: this.visualTool,
			visualCorrection: this.visualCorrectionTool,
			draw: this.basicTool,
		}
		const activateTool = (value) => {
			const t = toolByValue[value]
			if (t) t.activate()
		}
		this.shadow.querySelectorAll('input[name="pattern-tool"]').forEach((input) => {
			input.addEventListener("change", () => {
				if (input.checked) activateTool(input.value)
			})
		})
		const initialTool = this.shadow.querySelector('input[name="pattern-tool"]:checked')
		if (initialTool) activateTool(initialTool.value)

		this.shadow.querySelectorAll('input[name="pattern-model"]').forEach((input) => {
			input.addEventListener("change", () => {
				this.activeModelName = input.value
				this.socket.emit("change:model", input.value)
				console.log("model changed to", input.value)
			})
		})
	}


	sendLine(line) {
		let processedLine = this.canvas.processLine(line)
		this.socket.emit("add:line", processedLine)
		console.log("sent line", processedLine)
	}
	
}

customElements.define('pattern-tools', PatternTools);
