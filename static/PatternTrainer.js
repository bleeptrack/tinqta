'use strict';
import { SceneButton } from './SceneButton.js';
import { PaperCanvas } from './PaperCanvas.js';
import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js";

export class PatternTrainer extends HTMLElement {
	constructor(n) {
		
		super();
		this.socket = io();
		this.shadow = this.attachShadow({ mode: 'open' });
		this.canvas = new PaperCanvas()
		
		this.socket.on("init", (config) => {
			console.log("config received", config)
			this.canvas.setConfig(config)
		})
		
		this.socket.on("progress", (text) => {
			console.log("progress received", text)
		})
		
		

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				
				
			</style>
			
			<div id="container">
				<input type="test" placeholder="Scribble Name" id="name"></input>
				
				<button id="undo">undo</button>
				<button id="train">train</button>
			</div>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));

		this.shadow.getElementById("name").after(this.canvas)
		this.shadow.getElementById("undo").addEventListener("click", () => {
			this.canvas.undo()
		})
		this.shadow.getElementById("train").addEventListener("click", () => {
			console.log(this.canvas.linelist)
			console.log("name:", this.shadow.getElementById("name").value)
			
			this.socket.emit("new dataset", {
				name: this.shadow.getElementById("name").value,
				list: this.canvas.linelist,
			})
			
		})

	}


	connectedCallback() {
		
		
	}

}

customElements.define('pattern-trainer', PatternTrainer);
