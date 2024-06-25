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
					
					<button id="download">SAVE</button>
				</div>
				<div id="right">
				</div>
			</div>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));
		
		this.shadow.getElementById("download").addEventListener("click", this.downloadSVG.bind(this))
		
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
			this.progressbar.setPercentage(data.detail.percentage, data.detail.label)
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
