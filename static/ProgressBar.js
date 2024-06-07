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
					
				}
			</style>
			
			<div id="bar">
				<div id="title"></div>
				<div id="progress" class="scribble"></div>
			</div>
				
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));
		
		
		

	}
	


	connectedCallback() {
		this.setPercentage(1)
		
	}
	
	setPercentage(percentage){
		for(let rule of this.shadow.styleSheets[this.shadow.styleSheets.length-1].cssRules){
			console.log(rule.selectorText)
			if(rule.selectorText  == "#progress::after" ){
				console.log(rule);
				rule.style.width = `calc(${percentage}% - 1px)`
			}
		}
	}

}

customElements.define('progress-bar', ProgressBar);
