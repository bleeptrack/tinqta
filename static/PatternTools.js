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
		this.activeModelName = 'triangles'
		this.isApplyingNoise = false
		this.isSwitchingModel = false
		this.pendingToolValue = null
		this.basicTool = new paper.Tool()
		this.activeLine = null

		this.basicTool.onMouseDown = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			this.activeLine = new paper.Path()
			this.activeLine.strokeColor = "darkgray"
			this.activeLine.strokeWidth = 3
			this.activeLine.strokeCap = 'round'
			this.activeLine.add(event.point)
		}
		this.basicTool.onMouseDrag = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			this.activeLine.add(event.point)
		}
		this.basicTool.onMouseUp = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			this.activeLine.simplify()
			if(this.activeLine.segments.length > 1){
				this.sendLine(this.activeLine)
			}else{
				this.activeLine.remove()
			}
			this.activeLine = null
		}
		
		this.socket.on("init", (config) => {
			this.canvas.setConfig(config)
			
		})

		this.stampTool = new paper.Tool()
		this.stampTool.onMouseDown = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			this.activeLine = new paper.Path.Circle(event.point, 10)
			this.activeLine.fillColor = "green"
		}
		this.stampTool.onMouseUp = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			this.activeLine.remove()
			this.socket.emit("add:stamp", {"position": {"x": event.point.x, "y": event.point.y}})
		}

		/*
		this.stampCorrectionTool = new paper.Tool()
		this.stampCorrectionTool.onMouseUp = (event) => {
			this.socket.emit("add:stamp", {"position": {"x": event.point.x, "y": event.point.y}})
		}
		*/

		this.visualTool = new paper.Tool()
		this.visualTool.onMouseDown = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			this.activeLine = new paper.Path()
			this.activeLine.strokeColor = "blue"
			this.activeLine.strokeWidth = 3
			this.activeLine.strokeCap = 'round'
			this.activeLine.add(event.point)
		}
		this.visualTool.onMouseDrag = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			this.activeLine.add(event.point)
		}
		this.visualTool.onMouseUp = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			if (this.activeLine.segments.length <= 1) return
			this.activeLine.simplify()
			let processedLine = this.canvas.processLine(this.activeLine)
			this.socket.emit("add:visual", {"position": {"x": this.activeLine.firstSegment.point.x, "y": this.activeLine.firstSegment.point.y},"line": processedLine, "name": this.activeModelName, "correction": false})
			this.activeLine = null
		}

		/*
		this.visualCorrectionTool = new paper.Tool()
		this.visualCorrectionTool.onMouseDown = (event) => {
			this.activeLine = new paper.Path()
			this.activeLine.strokeColor = "blue"
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
		*/

		this.eraseTool = new paper.Tool()
		this.eraseTool.onMouseDown = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			this.activeLine = new paper.Path()
			this.activeLine.strokeColor = "red"
			this.activeLine.strokeWidth = 3
			this.activeLine.strokeCap = 'round'
			this.activeLine.add(event.point)
		}
		this.eraseTool.onMouseDrag = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			this.activeLine.add(event.point)
		}
		this.eraseTool.onMouseUp = (event) => {
			if (this.isSwitchingModel || this.isApplyingNoise) return
			let indices = []
			for(let [index, line] of this.canvas.linelist.entries()){
				if(line.intersects(this.activeLine)){
					
						indices.push(index)
					
				}
			}
			this.socket.emit("erase:lines", indices)
			this.activeLine.remove()
			this.activeLine = null
		}
		
		
		this.socket.on("init", (config) => {
			this.canvas.setConfig(config)
			
		})

		this.socket.on('toast', (data) => {
			console.info(`MESSAGE: ${data.message}`)
			
		})

		this.socket.on('set:info', (data) => {
			if (typeof data?.model === "string" && data.model.length > 0) {
				this.activeModelName = data.model
				const modelInputs = this.shadow.querySelectorAll('input[name="pattern-model"]')
				modelInputs.forEach((input) => {
					input.checked = input.value === this.activeModelName
				})
			}
			const correctionToggle = this.shadow.getElementById("correction-toggle")
			if (correctionToggle && typeof data?.correction === "boolean") {
				correctionToggle.checked = data.correction
			}
		})

		this.socket.on('draw:lines', (data) => {
			this.canvas.clear()
			this.canvas.linelist = []
			for(let line of data.lines){

				let l = this.canvas.drawLine(line, "black")
				l.strokeWidth = 3
				l.strokeCap = 'round'
				l.usedTool = line["added_at_stage"]
				this.canvas.linelist.push(l)

				/*
				let mark = l.clone()
				if(line["added_at_stage"]){
					if(line["added_at_stage"].includes("stamp")){
						mark.strokeColor = "green"
					}
					if(line["added_at_stage"].includes("visual")){
						mark.strokeColor = "blue"
					}
					if(line["added_at_stage"].includes("correction")){
						mark.dashArray = [5, 15]
					}
				}
				
				mark.strokeWidth = 10
				mark.strokeCap = 'round'
				mark.opacity = 0.5
				mark.model = line["added_with_model"]
				l.addChild(mark)
				*/

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
				#undo,
				#clear{
					position: absolute;
					top: 2vh;
					width: 2em;
				}
				#undo{
					right: 2vh;
				}
				#clear{
					right: calc(2vh + 2.5em);
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
					gap: 0.65rem;
					min-height: 0;
					padding-inline: 0.35rem;
					padding-bottom: 0.2rem;
					box-sizing: border-box;
					overflow: hidden;
				}
				#sidebar-content {
					flex: 1 1 auto;
					min-height: 0;
					min-width: 0;
					overflow-x: hidden;
					overflow-y: auto;
					display: flex;
					flex-direction: column;
					gap: 1.25rem;
				}
				#sidebar-footer {
					flex: 0 0 auto;
					min-width: 0;
					width: 100%;
					box-sizing: border-box;
					padding-inline: 0.15rem 0.55rem;
					display: flex;
					flex-direction: column;
					gap: 0.75rem;
				}
				#sidebar-footer #save-svg {
					width: 100%;
					max-width: none;
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
				#tools-container .tool-radios {
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
				label.toggle-row.scribble .toggle-row-icon {
					font-family: "Material Symbols Outlined";
					font-size: 1.05rem;
					line-height: 1;
					margin-right: 0.45rem;
					pointer-events: none;
				}
				label.toggle-row.scribble .toggle-row-content {
					display: inline-flex;
					align-items: center;
					justify-content: center;
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
				#model-container label.toggle-row.scribble {
					flex-direction: row;
					align-items: center;
					justify-content: flex-start;
					padding: 0.35rem 0.45rem 0.35rem 1.15rem;
					gap: 0;
					width: 100%;
				}
				#model-container .model-option-content {
					display: flex;
					flex-direction: row;
					align-items: center;
					justify-content: flex-start;
					gap: 0.55rem;
					flex: 1;
					min-width: 0;
					width: 100%;
				}
				#model-container .model-preview {
					display: block;
					flex: 0 0 58%;
					width: 58%;
					max-width: 58%;
					height: 8.5rem;
					object-fit: contain;
					object-position: left center;
					pointer-events: none;
					min-width: 0;
					margin-left: 0.65rem;
					margin-right: 0.2rem;
					box-sizing: border-box;
				}
				#model-container .toggle-row-label {
					flex: 1;
					display: flex;
					align-items: center;
					justify-content: center;
					text-align: center;
					min-width: 0;
				}
				#model-container.is-disabled {
					opacity: 0.5;
				}
				#model-container.is-disabled label {
					pointer-events: none;
					cursor: not-allowed;
				}
				#tools-container label.correction-check.scribble {
					display: flex;
					align-items: center;
					justify-content: center;
					gap: 0.5rem;
					margin-top: 0.35rem;
					cursor: pointer;
					user-select: none;
					width: auto;
					max-width: none;
					padding: 0.95rem 1.4rem;
					line-height: 1.2;
					background-color: #fff;
				}
				#tools-container label.correction-check.scribble.is-disabled {
					opacity: 0.5;
					cursor: not-allowed;
				}
				#tools-container label.correction-check.scribble::after {
					width: calc(100% - 6px);
					height: calc(100% - 6px);
					left: 3px;
					bottom: 3px;
				}
				#tools-container label.correction-check input {
					appearance: none;
					-webkit-appearance: none;
					width: 1rem;
					height: 1rem;
					margin: 0;
					cursor: pointer;
					border: 2px solid #000;
					background: #fff;
					border-radius: 2px;
					display: grid;
					place-content: center;
				}
				#tools-container label.correction-check input::before {
					content: "";
					width: 0.55rem;
					height: 0.55rem;
					transform: scale(0);
					transition: transform 120ms ease-in-out;
					background: #000;
					clip-path: polygon(14% 44%, 0 65%, 45% 100%, 100% 16%, 80% 0, 43% 62%);
				}
				#tools-container label.correction-check input:checked::before {
					transform: scale(1);
				}
				#tools-container label.correction-check input:checked {
					background: var(--main-color);
					border-color: var(--main-color);
				}
				#tools-container label.correction-check input:checked::before {
					background: #fff;
				}
				#tools-container label.correction-check input:focus-visible {
					outline: 2px solid #000;
					outline-offset: 2px;
				}
				.noise-row {
					display: flex;
					flex-direction: column;
					gap: 0.45rem;
					border: 2px solid #000;
					background: #fff;
					padding: 0.6rem 0.7rem;
				}
				.noise-row .noise-label {
					display: block;
					font-weight: 700;
					letter-spacing: 0.02em;
				}
				.noise-row .noise-controls {
					display: flex;
					align-items: center;
					gap: 0.5rem;
					flex-wrap: nowrap;
				}
				.noise-row input[type="range"] {
					flex: 1;
					min-width: 0;
					height: auto;
					padding: 0;
				}
				.noise-row .noise-value {
					min-width: 2.25rem;
					font-variant-numeric: tabular-nums;
					font-weight: 700;
					padding: 0.15rem 0.4rem;
					border: 2px solid #000;
					background: #fff;
					text-align: center;
				}
				.noise-row button.apply-noise.scribble {
					width: auto;
					max-width: none;
					flex: 0 0 auto;
					padding: 0.35rem 0.95rem;
					line-height: 1.2;
				}
				.noise-row button.apply-noise.scribble::after {
					left: 2px;
					bottom: 2px;
					width: calc(100% - 4px);
					height: calc(100% - 4px);
				}
			</style>
			
			<div id="container">

				<!-- <div id="name" class="scribble input" placeholder="Enter your model name" contenteditable=true></div> -->
				
				<div id="canvas-area">
					<div id="canvas-container">
						<button class="scribble material-symbols-outlined" id="clear">delete</button>
						<button id="undo" class="material-symbols-outlined scribble">undo</button>
					</div>
				</div>
				<aside id="sidebar">
					<div id="sidebar-content">
					<section class="button-section" aria-label="Tools">
						<h2 id="tools-heading">Tools</h2>
						<div id="tools-container">
							<div class="tool-radios" role="radiogroup" aria-labelledby="tools-heading">
								<label class="scribble toggle-row"><input type="radio" name="pattern-tool" value="stamp"><span class="toggle-row-content"><span class="toggle-row-icon" aria-hidden="true">ads_click</span><span class="toggle-row-label">stamp</span></span></label>
								<label class="scribble toggle-row"><input type="radio" name="pattern-tool" value="visual"><span class="toggle-row-content"><span class="toggle-row-icon" aria-hidden="true">shape_line</span><span class="toggle-row-label">adapt</span></span></label>
								<label class="scribble toggle-row"><input type="radio" name="pattern-tool" value="draw" checked><span class="toggle-row-content"><span class="toggle-row-icon" aria-hidden="true">draw</span><span class="toggle-row-label">draw</span></span></label>
								<label class="scribble toggle-row"><input type="radio" name="pattern-tool" value="erase"><span class="toggle-row-content"><span class="toggle-row-icon" aria-hidden="true">ink_eraser</span><span class="toggle-row-label">erase</span></span></label>
								<label id="correction-controls" class="correction-check scribble">
									<input type="checkbox" id="correction-toggle" name="correction-toggle">
									<span>Correction</span>
								</label>
								
								
							</div>
						</div>
					</section>
					<section class="button-section" aria-label="Models">
						<h2 id="models-heading">Models</h2>
						<div id="model-container" role="radiogroup" aria-labelledby="models-heading">
							<label class="scribble toggle-row"><input type="radio" name="pattern-model" value="boxes"><span class="model-option-content"><img class="model-preview" src="/static/boxes.svg" alt=""><span class="toggle-row-label">boxes</span></span></label>
							<label class="scribble toggle-row"><input type="radio" name="pattern-model" value="swirls"><span class="model-option-content"><img class="model-preview" src="/static/swirls.svg" alt=""><span class="toggle-row-label">swirls</span></span></label>
							<label class="scribble toggle-row"><input type="radio" name="pattern-model" value="triangles" checked><span class="model-option-content"><img class="model-preview" src="/static/triangles.svg" alt=""><span class="toggle-row-label">triangles</span></span></label>
						</div>
					</section>
					</div>
					<div id="sidebar-footer">
						<div class="noise-row">
							<span class="noise-label">Noise (lines)</span>
							<div class="noise-controls">
								<input type="range" id="line-noise-slider" min="0" max="0.05" step="0.001" value="0.01">
								<span class="noise-value" id="line-noise-value" aria-live="polite">0.01</span>
								<button type="button" class="apply-noise scribble" id="line-noise-apply">Apply</button>
							</div>
						</div>
						<button type="button" class="scribble" id="save-svg" aria-label="Save drawing as SVG"><span class="material-symbols-outlined">save</span>SUBMIT</button>
					</div>
				</aside>
			</div>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));

		this.shadow.getElementById("canvas-container").appendChild(this.canvas)

		const toolByValue = {
			stamp: this.stampTool,
			stampCorrection: this.stampCorrectionTool,
			visual: this.visualTool,
			visualCorrection: this.visualCorrectionTool,
			draw: this.basicTool,
			erase: this.eraseTool,
		}
		const activateTool = (value) => {
			const t = toolByValue[value]
			if (t) t.activate()
		}
		const modelContainer = this.shadow.getElementById("model-container")
		const toolInputs = () => this.shadow.querySelectorAll('input[name="pattern-tool"]')
		const modelInputs = () => this.shadow.querySelectorAll('input[name="pattern-model"]')
		const setModelSwitchBusy = (isBusy) => {
			this.isSwitchingModel = isBusy
			if (isBusy && this.activeLine) {
				try {
					this.activeLine.remove()
				} catch (e) {
					// ignore cleanup failures; Paper.js state can vary mid-gesture
				}
				this.activeLine = null
			}
			toolInputs().forEach((input) => {
				input.disabled = isBusy
			})
			modelInputs().forEach((input) => {
				input.disabled = isBusy
			})
			modelContainer.classList.toggle("is-disabled", isBusy)
		}
		const correctionControls = this.shadow.getElementById("correction-controls")
		const correctionToggle = this.shadow.getElementById("correction-toggle")
		const setAdvancedControlsEnabled = (toolValue) => {
			const enabledTools = new Set(["stamp", "visual", "stampCorrection", "visualCorrection"])
			const isEnabled = enabledTools.has(toolValue)
			const correctionEnabled = toolValue === "stamp"
				|| toolValue === "stampCorrection"
				|| toolValue === "visual"
				|| toolValue === "visualCorrection"
			correctionToggle.disabled = !correctionEnabled
			correctionControls.classList.toggle("is-disabled", !correctionEnabled)
			modelContainer.classList.toggle("is-disabled", !isEnabled)
			this.shadow.querySelectorAll('input[name="pattern-model"]').forEach((input) => {
				input.disabled = !isEnabled
			})
		}
		toolInputs().forEach((input) => {
			input.addEventListener("change", () => {
				if (input.checked) {
					if (this.isSwitchingModel) {
						this.pendingToolValue = input.value
						return
					}
					activateTool(input.value)
					setAdvancedControlsEnabled(input.value)
				}
			})
		})
		const initialTool = this.shadow.querySelector('input[name="pattern-tool"]:checked')
		if (initialTool) {
			activateTool(initialTool.value)
			setAdvancedControlsEnabled(initialTool.value)
		}

		modelInputs().forEach((input) => {
			input.addEventListener("change", () => {
				this.activeModelName = input.value
				setModelSwitchBusy(true)
				this.socket.emit("change:model", { name: input.value }, () => {
					setModelSwitchBusy(false)
					const toolToActivate = this.pendingToolValue ?? this.shadow.querySelector('input[name="pattern-tool"]:checked')?.value
					this.pendingToolValue = null
					if (toolToActivate) {
						activateTool(toolToActivate)
						setAdvancedControlsEnabled(toolToActivate)
					}
				})
			})
		})

		correctionToggle.addEventListener("change", () => {
			this.socket.emit("change:correction", { correction: correctionToggle.checked })
		})
		this.socket.on("correctionChanged", (payload) => {
			if (typeof payload?.correction === "boolean") {
				correctionToggle.checked = payload.correction
			}
		})

		const noiseSlider = this.shadow.getElementById("line-noise-slider")
		const noiseValue = this.shadow.getElementById("line-noise-value")
		const noiseApplyButton = this.shadow.getElementById("line-noise-apply")
		const setNoiseApplyBusy = (isBusy) => {
			this.isApplyingNoise = isBusy
			noiseApplyButton.disabled = isBusy
			noiseApplyButton.textContent = isBusy ? "Applying..." : "Apply"
		}
		const syncNoiseLabel = () => {
			const v = Number(noiseSlider.value)
			noiseValue.textContent = Number.isFinite(v) ? v.toFixed(3) : noiseSlider.value
			noiseSlider.setAttribute("aria-valuenow", noiseValue.textContent)
		}
		noiseSlider.addEventListener("input", syncNoiseLabel)
		syncNoiseLabel()
		noiseApplyButton.addEventListener("click", () => {
			if (this.isApplyingNoise) {
				return
			}
			const amount = parseFloat(noiseSlider.value)
			setNoiseApplyBusy(true)
			this.socket.emit("apply:noise", { noise_level: amount }, () => {
				setNoiseApplyBusy(false)
			})
		})

		this.shadow.getElementById("clear").addEventListener("click", () => {
			if (window.confirm("Are you sure you want to clear the canvas?")) {
				this.socket.emit('clear')
			}
		})

		this.shadow.getElementById("undo").addEventListener("click", () => {
			this.socket.emit('undo')
		})

		this.shadow.getElementById("save-svg").addEventListener("click", () => {
			this.saveSVG()
		})
	}


	sendLine(line) {
		let processedLine = this.canvas.processLine(line)
		this.socket.emit("add:line", processedLine)
	}

	async saveSVG(){
		this.canvas.colorByTool()
		const svg = paper.project.exportSVG({ asString: true })
		try {
			const response = await fetch('/save-svg', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					svg,
					model: this.activeModelName,
				}),
			})
			if (!response.ok) {
				throw new Error(`Failed to save SVG (${response.status})`)
			}
			const result = await response.json()
			window.alert(`Saved successfully. Thanks :)`)
			this.socket.emit('clear')
		} catch (error) {
			console.error('Error saving SVG:', error)
			window.alert('Failed to save drawing. Please try again.')
		}
	}
	
}

customElements.define('pattern-tools', PatternTools);
