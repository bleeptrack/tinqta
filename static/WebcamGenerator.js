'use strict';
import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js";
import { VectorizerCanvas } from './VectorizerCanvas.js';

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
				
				
			</style>
			
			<div id="container">
				hello hello
				<button id="download">SAVE</button>
				<vectorizer-canvas id="vec"></vectorizer-canvas>
			</div>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));
		
		this.shadow.getElementById("download").addEventListener("click", this.downloadSVG.bind(this))
	}


	connectedCallback() {
		
		
	}
	
	downloadSVG(){
		var svg = this.shadow.getElementById("vec").getSVG()
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
