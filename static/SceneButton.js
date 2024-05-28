'use strict';

export class SceneButton extends HTMLElement {
	constructor(name, path, icon) {
		
		super();
		this.name = name
		this.path = path
		this.icon = icon
		this.shadow = this.attachShadow({ mode: 'open' });

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			<link rel="stylesheet" href="/static/style.css">
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				
				div{
					background-color: gray;
					width: 100%;
				}
				
				.active{
					background-color: red;
				}
				
				.largeicon{
					font-size: 5em !important;
				}
				
				
					
					button{
						display: flex;
						justify-content: space-around;
						align-items: center;
					}

					@media (min-width: 768px) {
						.scribble {
							padding: .75rem 3rem;
							font-size: 1.25rem;
						}
					}
				
			</style>
			
			<button id="${this.name}" class="scribble"><span class="material-symbols-outlined largeicon">${this.icon}</span>${this.name}</button>
				
			
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));

		this.shadow.getElementById(this.name).style.transform = `rotate(${Math.random() * 10 -5}deg)`;
		
		this.shadow.getElementById(this.name).addEventListener("click", () => {
			window.location.assign(this.path)
		})

	}
	


	connectedCallback() {
		
	}

}

customElements.define('scene-button', SceneButton);
