'use strict';

export class ProgressBar extends HTMLElement {
	constructor(n) {
		
		super();
	
		this.shadow = this.attachShadow({ mode: 'open' });

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			<link rel="stylesheet" href="/static/style.css">
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				
				#bar{
					width: 100%;
				}
				#progress{
					height: 10vh;
					
				}
				#progress:after{
					background-color: rgba(1,1,1,0.5);
					width: calc(1% - 1px) !important;
				}
			</style>
			
			<div id="bar">
				<div id="title"></div>
				<div id="progress" class="scribble"></div>
			</div>
				
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));
		this.setPercentage(50)

	}
	


	connectedCallback() {
		
	}
	
	setPercentage(percentage){
		this.shadow.getElementById("progress").style.width = `width: calc(${percentage}% - 1px) !important;`
	}

}

customElements.define('progress-bar', ProgressBar);
