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
				
				.scribble{
					background-color: var(--main-color);
					border: 0 solid #E5E7EB;
					box-sizing: border-box;
					color: #000000;
					font-family: ui-sans-serif,system-ui,-apple-system,system-ui,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji";
					font-size: 1rem;
					font-weight: 700;
					justify-content: center;
					line-height: 1.75rem;
					padding: .75rem 1.65rem;
					position: relative;
					text-align: center;
					text-decoration: none #000000 solid;
					text-decoration-thickness: auto;
					width: 100%;
					max-width: 460px;
					position: relative;
					cursor: pointer;
					user-select: none;
					-webkit-user-select: none;
					touch-action: manipulation;
					}

					.scribble:focus {
					outline: 0;
					}

					.scribble:after {
					content: '';
					position: absolute;
					border: 2px solid #000000;
					bottom: 4px;
					left: 4px;
					width: calc(100% - 1px);
					height: calc(100% - 1px);
					}

					.scribble:hover:after {
					bottom: 2px;
					left: 2px;
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
