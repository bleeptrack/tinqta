'use strict';
import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js";
import { VectorizerCanvas } from './VectorizerCanvas.js';
import { ProgressBar } from './ProgressBar.js';

export class WebcamGenerator extends HTMLElement {
	constructor(n) {
		
		super();
		this.socket = io();
		this.shadow = this.attachShadow({ mode: 'open' });
		
		
		this.socket.on("init", (config) => {
			console.log("config received", config)
		})
		this.socket.on("models", (models) => {
			console.log("models received", models)
			let select = this.shadow.getElementById("model")
			for(let modelname of models){
				let option = document.createElement("option")
				option.value = modelname
				option.innerHTML = modelname
				select.appendChild(option)
			}
		})
		
		
		

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			<link rel="stylesheet" href="/static/style.css">
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				
				#container{
					width: 100%;
					height: 100%;
					display: flex;
					gap: 10vh;
					
				}
				#left{
					width: 40vw;
					height: 100%;
					backgroundColor: grey;
					display: flex;
					flex-direction: column;
					flex-grow: 1;
					align-items: center;
				}
				#right{
					width: 30vw;
					height: 100%;
				}
				progress-bar{
					width: 100%;
				}
			</style>
			
			
			<div id="container">
				<div id="left">
					<vectorizer-canvas id="vec"></vectorizer-canvas>
				</div>
				<div id="right">
					<input type="range" min="1" max="100" value="20" class="slider" id="edge-min">
					<input type="range" min="1" max="100" value="30" class="slider" id="edge-max">
					<select name="model" id="model">
						<option value="random">random</option>
					</select>
				</div>
			</div>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));
		
		this.vectorizer = this.shadow.getElementById("vec")
		this.vectorizer.addEventListener("ready", () => {
			let startBtn = document.createElement("button")
			startBtn.id = "start"
			startBtn.classList.add("scribble")
			startBtn.innerHTML = "START"
			this.vectorizer.after(startBtn)
			
			startBtn.addEventListener("click", () => {
				this.vectorizer.startProcess()
				this.progressbar = new ProgressBar()
				startBtn.replaceWith(this.progressbar)
			})
		})
		this.vectorizer.addEventListener("progress", (data) => {
			if(data.detail.percentage == 100){
				let saveBtn = document.createElement("button")
				saveBtn.addEventListener("click", this.downloadSVG.bind(this))
				saveBtn.classList.add("scribble")
				saveBtn.innerHTML = "SAVE"
				this.progressbar.replaceWith(saveBtn)
			}else{
				this.progressbar.setPercentage(data.detail.percentage, data.detail.label)
			}
		})
		
		this.shadow.getElementById("edge-min").addEventListener("change", (event) => {
			this.vectorizer.edgemin = event.target.value
		})
		
		this.shadow.getElementById("edge-max").addEventListener("change", (event) => {
			this.vectorizer.edgemax = event.target.value
		})
		
		this.shadow.getElementById("model").addEventListener("change", (event) => {
			console.log(event.target.value)
			this.vectorizer.setModelName(event.target.value)
		})
		
		
		
	}


	connectedCallback() {
		
		
	}
	
	downloadSVG(){
		var svg = this.vectorizer.getSVG()
		var svgBlob = new Blob([svg], {type:"image/svg+xml;charset=utf-8"});
		var svgUrl = URL.createObjectURL(svgBlob);
		var downloadLink = document.createElement("a");
		downloadLink.href = svgUrl;
		downloadLink.download = "stamp.svg";
		document.body.appendChild(downloadLink);
		downloadLink.click();
		document.body.removeChild(downloadLink);
	}

}

customElements.define('webcam-generator', WebcamGenerator);
