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
			
		})
		
		this.socket.on("progress", (text) => {
			console.log("progress received", text)
			if(text.lines){				
				this.canvas.trainingEpoch(text)
				this.epochLines.push(text)
				
			}else if(text.percent){
				this.progressbar.setPercentage(Number(text.percent), text.label)
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
				
			</style>
			
			<div id="container">
				<h1>Train your Scribble Model</h1>
				<div id="name" class="scribble input" placeholder="Enter your model name" contenteditable=true></div>
				<div id="canvas-container">
					<button id="undo" class="material-symbols-outlined scribble">undo</button>
				</div>
				
				<button id="train" class="scribble">train</button>
				<button id="train-pattern" class="scribble">train pattern</button>
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
		
	}


	connectedCallback() {
		window.downloadLines = this.saveLines.bind(this)
		this.canvas.setPlaceholder()
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
