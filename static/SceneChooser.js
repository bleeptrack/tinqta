'use strict';
import { SceneButton } from './SceneButton.js';

export class SceneChooser extends HTMLElement {
	constructor() {
		
		super();
		this.shadow = this.attachShadow({ mode: 'open' });

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			
			<style>
				
				
				
			</style>
			
			<div id="container"></div>
				
			
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));

		this.shadow.getElementById("container").appendChild( new SceneButton("Train Model", "/train", "model_training") )

	}
	


	connectedCallback() {
		
	}

}

customElements.define('scene-chooser', SceneChooser);
