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
			let ul = this.shadow.getElementById("model-list");
			if(ul){
				ul.innerHTML = '';
				for(let modelname of models){
					let li = document.createElement('li');
					li.dataset.value = modelname;
					li.innerHTML = `
						<span>${modelname}</span>
						<button class="delete-option">X</button>
					`;
					ul.appendChild(li);
				}
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
					flex-direction: column;
					justify-content: space-around;
					padding: 3vw;
					box-sizing: border-box; 
				}
				#wrapper{
					border: 5px dashed grey;
					
					position: relative;
					width: 100%;
					height: 100%;
					padding: 5px;
				}
				#window{
					width: 100%;
					height: 100%;
					overflow: scroll;
					display: flex;
					flex-direction: column;
					flex-grow: 1;
					align-items: center;
					justify-content: center;
					scrollbar-width: none;
				}
				#settings{
					position: absolute;
					bottom: 0;
					left: 0;
					background: rgba(255,255,255,0.5);
					width: 100%;
					height: 20%;
					flex-direction: column;
					justify-content: space-evenly;
					align-items: center;
					display: none;
					font-family: 'monospace';
				}
				#right{
					background-color: grey;
				}
				progress-bar{
					width: 95%;
				}
				
				svg{
					width: 80%;
					height: auto;
				}
				

				.custom-dropdown{
					display: flex;
					justify-content: center;
					align-items: center;
					margin-top: 2vh;
				}
				
				@keyframes dash {
					100% { stroke-dashoffset: 0; }
				}
			</style>
			
			
			<div id="container">
				<div id="wrapper">
					<div id="window">
						<vectorizer-canvas id="vec"></vectorizer-canvas>
					</div>
					<div id="settings">
						<p id="settings-description">
							Adjust the edge detection settings to fine-tune the vectorization process. 
							Adjust min and max thresholds to control edge sensitivity.
						</p>
						<div id="settings-edge">
							<label for="edge-detail">Edge Detail:</label>
							<input type="range" min="1" max="8" value="2" class="slider" id="edge-detail">
							<label for="edge-min">Edge Min:</label>
							<input type="range" min="1" max="100" value="20" class="slider" id="edge-min">
							<label for="edge-max">Edge Max:</label>
							<input type="range" min="1" max="100" value="30" class="slider" id="edge-max">
							<div class="custom-dropdown">
								<button id="dropdown-toggle" class="scribble">Select Model</button>
								<div id="dropdown-popover" popover>
									<ul id="model-list">
										<li data-value="random">
											<span>random</span>
										</li>
									</ul>
									<button id="add-model" class="scribble">Add Model</button>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));

		const toggle = this.shadow.getElementById('dropdown-toggle');
		const popover = this.shadow.getElementById('dropdown-popover');
		const modelList = this.shadow.getElementById('model-list');
		const addModelBtn = this.shadow.getElementById('add-model');

		toggle.addEventListener('click', () => popover.togglePopover());

		modelList.addEventListener('click', (e) => {
			if (e.target.classList.contains('delete-option')) {
				const li = e.target.closest('li');
				const modelName = li.dataset.value;
				if (confirm(`Are you sure you want to delete the model "${modelName}"?`)) {
					this.socket.emit('deleteModel', { modelName });
					li.remove();
				}
			} else if (e.target.tagName === 'SPAN') {
				toggle.textContent = e.target.textContent;
				this.vectorizer.setModelName(e.target.textContent)
				this.shadow.getElementById("start").disabled = false
				popover.hidePopover();
			}
		});

		addModelBtn.addEventListener('click', () => {
			window.open('/train', '_self')
		})


		
		this.vectorizer = this.shadow.getElementById("vec")
		this.vectorizer.addEventListener("ready", () => {
			let startBtn = document.createElement("button")
			startBtn.id = "start"
			startBtn.classList.add("scribble")
			startBtn.innerHTML = "START"
			startBtn.disabled = true
			this.shadow.getElementById("settings").appendChild(startBtn)
			
			startBtn.addEventListener("click", () => {
				this.vectorizer.startProcess()
				this.progressbar = new ProgressBar()
				startBtn.replaceWith(this.progressbar)
				this.shadow.getElementById("settings-edge").style.display = "none"
			})

			this.shadow.getElementById("settings").style.display = "flex"
		})
		this.vectorizer.addEventListener("progress", (data) => {
			if(data.detail.percentage == 100){

				this.svg = this.vectorizer.getSVG(false)
				this.svg.querySelectorAll("path").forEach( p => {
					p.style.strokeDasharray = p.getTotalLength() 
					p.style.strokeDashoffset = p.getTotalLength() 
				})
				this.shadow.getElementById("vec").replaceWith(this.svg)

				this.shadow.getElementById("settings-description").textContent = "Adjust colors, style and animation length. Download SVG or copy the HTML Code to your clipboard."

				let settingsEdge = this.shadow.getElementById("settings-edge")
				settingsEdge.style.display = "flex"
				settingsEdge.innerHTML = ""

				// Create color pickers for stroke and shadow
				let strokeColorPicker = document.createElement("input");
				strokeColorPicker.type = "color";
				strokeColorPicker.id = "stroke-color";
				strokeColorPicker.value = sessionStorage.getItem('tinqta:strokeColor') || "black"; // Default to black
				
				let shadowColorPicker = document.createElement("input");
				shadowColorPicker.type = "color";
				shadowColorPicker.id = "shadow-color";
				shadowColorPicker.value = sessionStorage.getItem('tinqta:shadowColor') || "teal"; // Default to white
				
				// Create labels for the color pickers
				let strokeLabel = document.createElement("label");
				strokeLabel.htmlFor = "stroke-color";
				strokeLabel.textContent = "Stroke Color:";
				
				let shadowLabel = document.createElement("label");
				shadowLabel.htmlFor = "shadow-color";
				shadowLabel.textContent = "Shadow Color:";
				
				// Create containers for each color picker and its label
				let strokeContainer = document.createElement("div");
				strokeContainer.appendChild(strokeLabel);
				strokeContainer.appendChild(strokeColorPicker);
				
				let shadowContainer = document.createElement("div");
				shadowContainer.appendChild(shadowLabel);
				shadowContainer.appendChild(shadowColorPicker);
				
				// Add the containers to the settingsEdge
				settingsEdge.appendChild(strokeContainer);
				settingsEdge.appendChild(shadowContainer);
				
				// Add event listeners to update SVG colors
				strokeColorPicker.addEventListener("input", updateSVGColors.bind(this));
				shadowColorPicker.addEventListener("input", updateSVGColors.bind(this));
				
				function updateSVGColors() {
					sessionStorage.setItem('tinqta:strokeColor', strokeColorPicker.value);
					sessionStorage.setItem('tinqta:shadowColor', shadowColorPicker.value);

					let strokes = this.svg.querySelector("g:first-child")
					let shadows = this.svg.querySelector("g:nth-child(2)")

					strokes.querySelectorAll("path").forEach(path => {
						path.style.stroke = strokeColorPicker.value;
					});

					shadows.querySelectorAll("path").forEach(path => {
						path.style.stroke = shadowColorPicker.value;
					});
				}

				// Create sliders for line thickness
				let strokeWidthSlider = document.createElement("input");
				strokeWidthSlider.type = "range";
				strokeWidthSlider.id = "stroke-width";
				strokeWidthSlider.min = "1";
				strokeWidthSlider.max = "10";
				strokeWidthSlider.value = sessionStorage.getItem('tinqta:strokeWidth') || "3";

				let shadowWidthSlider = document.createElement("input");
				shadowWidthSlider.type = "range";
				shadowWidthSlider.id = "shadow-width";
				shadowWidthSlider.min = "1";
				shadowWidthSlider.max = "10";
				shadowWidthSlider.value = sessionStorage.getItem('tinqta:shadowWidth') || "5";

				// Create sliders for opacity
				let strokeOpacitySlider = document.createElement("input");
				strokeOpacitySlider.type = "range";
				strokeOpacitySlider.id = "stroke-opacity";
				strokeOpacitySlider.min = "0";
				strokeOpacitySlider.max = "1";
				strokeOpacitySlider.step = "0.1";
				strokeOpacitySlider.value = sessionStorage.getItem('tinqta:strokeOpacity') || "1";

				let shadowOpacitySlider = document.createElement("input");
				shadowOpacitySlider.type = "range";
				shadowOpacitySlider.id = "shadow-opacity";
				shadowOpacitySlider.min = "0";
				shadowOpacitySlider.max = "1";
				shadowOpacitySlider.step = "0.1";
				shadowOpacitySlider.value = sessionStorage.getItem('tinqta:shadowOpacity') || "0.5";

				// Create slider for animation length
				let animationLengthSlider = document.createElement("input");
				animationLengthSlider.type = "range";
				animationLengthSlider.id = "animation-length";
				animationLengthSlider.min = "0";
				animationLengthSlider.max = "20";
				animationLengthSlider.value = sessionStorage.getItem('tinqta:animationLength') || "8";

				// Create labels for the sliders
				let strokeWidthLabel = document.createElement("label");
				strokeWidthLabel.htmlFor = "stroke-width";
				strokeWidthLabel.textContent = "Stroke Width:";

				let shadowWidthLabel = document.createElement("label");
				shadowWidthLabel.htmlFor = "shadow-width";
				shadowWidthLabel.textContent = "Shadow Width:";

				let strokeOpacityLabel = document.createElement("label");
				strokeOpacityLabel.htmlFor = "stroke-opacity";
				strokeOpacityLabel.textContent = "Stroke Opacity:";

				let shadowOpacityLabel = document.createElement("label");
				shadowOpacityLabel.htmlFor = "shadow-opacity";
				shadowOpacityLabel.textContent = "Shadow Opacity:";

				let animationLengthLabel = document.createElement("label");
				animationLengthLabel.htmlFor = "animation-length";
				animationLengthLabel.textContent = "Animation Length (s):";

				// Create containers for each slider and its label
				let strokeWidthContainer = document.createElement("div");
				strokeWidthContainer.appendChild(strokeWidthLabel);
				strokeWidthContainer.appendChild(strokeWidthSlider);

				let shadowWidthContainer = document.createElement("div");
				shadowWidthContainer.appendChild(shadowWidthLabel);
				shadowWidthContainer.appendChild(shadowWidthSlider);

				let strokeOpacityContainer = document.createElement("div");
				strokeOpacityContainer.appendChild(strokeOpacityLabel);
				strokeOpacityContainer.appendChild(strokeOpacitySlider);

				let shadowOpacityContainer = document.createElement("div");
				shadowOpacityContainer.appendChild(shadowOpacityLabel);
				shadowOpacityContainer.appendChild(shadowOpacitySlider);

				let animationLengthContainer = document.createElement("div");
				animationLengthContainer.appendChild(animationLengthLabel);
				animationLengthContainer.appendChild(animationLengthSlider);

				// Add the containers to the settingsEdge
				settingsEdge.appendChild(strokeWidthContainer);
				settingsEdge.appendChild(shadowWidthContainer);
				settingsEdge.appendChild(strokeOpacityContainer);
				settingsEdge.appendChild(shadowOpacityContainer);
				settingsEdge.appendChild(animationLengthContainer);

				// Add event listeners to update SVG styles
				[strokeWidthSlider, shadowWidthSlider, strokeOpacitySlider, shadowOpacitySlider].forEach(slider => {
					slider.addEventListener("input", updateSVGStyles.bind(this));
				});

				//retrigger animation
				animationLengthSlider.addEventListener("input", updateAnimation.bind(this));

				function updateAnimation(){
					sessionStorage.setItem('tinqta:animationLength', animationLengthSlider.value);

					const paths = this.svg.querySelectorAll("path");
					paths.forEach(path => {
						path.style.animation = 'none';
						path.offsetHeight; // Trigger reflow
						path.style.strokeDasharray = path.getTotalLength()
						path.style.strokeDashoffset = path.getTotalLength()
						path.style.animation = `dash ${animationLengthSlider.value}s ease-in-out forwards`;
					});
				}

				function updateSVGStyles() {
					// Save strokeWidthSlider value in local storage
					sessionStorage.setItem('tinqta:strokeWidth', strokeWidthSlider.value);
					sessionStorage.setItem('tinqta:shadowWidth', shadowWidthSlider.value);
					sessionStorage.setItem('tinqta:strokeOpacity', strokeOpacitySlider.value);
					sessionStorage.setItem('tinqta:shadowOpacity', shadowOpacitySlider.value);
					
					
					let strokes = this.svg.querySelector("g:first-child")
					let shadows = this.svg.querySelector("g:nth-child(2)")
					
					strokes.querySelectorAll("path").forEach(path => {
						path.setAttribute('vector-effect', 'non-scaling-stroke');
						path.style.strokeWidth = strokeWidthSlider.value;
						path.style.strokeOpacity = strokeOpacitySlider.value;
						path.style.fill = "none"
						//path.style.animation = `dash ${animationLengthSlider.value}s ease-in-out forwards`;
						path.style.strokeLinecap = "round"
					})
						
					shadows.querySelectorAll("path").forEach(path => {
						path.style.strokeWidth = shadowWidthSlider.value;
						path.style.strokeOpacity = shadowOpacitySlider.value;
						//path.style.animation = `dash ${animationLengthSlider.value}s ease-in-out forwards`;
						path.style.strokeLinecap = "round"
					})
				}

				updateSVGColors.bind(this)()
				updateSVGStyles.bind(this)()
				updateAnimation.bind(this)()

				let saveBtn = document.createElement("button")
				saveBtn.addEventListener("click", this.saveSVG.bind(this))
				saveBtn.classList.add("scribble")
				saveBtn.innerHTML = "SAVE"
				this.progressbar.replaceWith(saveBtn)

				// Create file picker for SVG branding
				let brandingPicker = document.createElement("input");
				brandingPicker.type = "file";
				brandingPicker.id = "branding-picker";
				brandingPicker.accept = ".svg";
				brandingPicker.style.display = "none";
				let brandingLabel = document.createElement("label");
				brandingLabel.htmlFor = "branding-picker";
				brandingLabel.classList.add("scribble");
				brandingLabel.innerHTML = "Choose optional branding SVG";

				saveBtn.insertAdjacentElement('beforebegin', brandingPicker);	
				saveBtn.insertAdjacentElement('beforebegin', brandingLabel);	

				brandingPicker.addEventListener("change", (event) => {
					const file = event.target.files[0];
					if (file) {
						const reader = new FileReader();
						reader.onload = (e) => {
						const brandingSVG = new DOMParser().parseFromString(e.target.result, 'image/svg+xml').documentElement;
						console.log(brandingSVG)
						// Get the main SVG element
						
						let strokes = this.svg.querySelector("g:first-child")
						
						// Create a new group for the branding
						const brandingGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
						brandingGroup.setAttribute("id", "branding");
						
						// Append all paths from the branding SVG to the new group
						const firstGroup = brandingSVG.querySelector('g');
						if (firstGroup) {
							brandingGroup.appendChild(firstGroup.cloneNode(true));
							let sibling = firstGroup.nextElementSibling;
							while (sibling) {
								brandingGroup.appendChild(sibling.cloneNode(true));
								sibling = sibling.nextElementSibling;
							}
						} else {
							console.warn('No group element found in the branding SVG');
						}
						
						// Append the branding group to the main SVG
						strokes.appendChild(brandingGroup);
						
						// Position the branding in the bottom right corner
						const svgBBox = this.svg.getBBox();
						const brandingBBox = brandingGroup.getBBox();
						console.log(brandingBBox)

						let minSize = Math.min(svgBBox.width, svgBBox.height)
						let scale = minSize * 0.3 / Math.max(brandingBBox.width, brandingBBox.height)
						//const scale = Math.min(svgBBox.width * 0.2 / brandingBBox.width, svgBBox.height * 0.2 / brandingBBox.height);
						//const translateX = svgBBox.width - (brandingBBox.width * scale) - 10;
						//const translateY = svgBBox.height + 10; // Position below the SVG
						
						brandingGroup.setAttribute('transform', `translate(${brandingBBox.x + minSize/20 + brandingBBox.width/2}, ${brandingBBox.y + minSize/20 + brandingBBox.height/2}) scale(${scale})`);
						
						// Store the branding SVG for later use
						this.vectorizer.brandingSVG = brandingGroup;
						
						// Update the SVG display
						updateSVGColors.bind(this)()
						updateSVGStyles.bind(this)()
						updateAnimation.bind(this)()
						};
						reader.readAsText(file);
					}
				});

				

				let copyBtn = document.createElement("button");
				copyBtn.addEventListener("click", this.copySVGToClipboard.bind(this));
				copyBtn.classList.add("scribble");
				copyBtn.innerHTML = '<span class="material-symbols-outlined">content_copy</span>';
				saveBtn.insertAdjacentElement('afterend', copyBtn);				
				
				
				
			}else{
				this.progressbar.setPercentage(data.detail.percentage, data.detail.label)
			}
		})
		
		
		
		
		
		
		this.vectorizer.edgemin = sessionStorage.getItem("tinqta:edge-min") || 20
		this.vectorizer.edgemax = sessionStorage.getItem("tinqta:edge-max") || 50
		this.vectorizer.edgeDetails = sessionStorage.getItem("tinqta:edge-detail") || 2
		this.vectorizer.brandingSVG = this.shadow.getElementById("tinqta:brandingSVG") || null

		this.shadow.getElementById("edge-min").value = this.vectorizer.edgemin
		this.shadow.getElementById("edge-max").value = this.vectorizer.edgemax
		this.shadow.getElementById("edge-detail").value = this.vectorizer.edgeDetails

		 this.shadow.getElementById("edge-min").addEventListener("change", (event) => {
			this.vectorizer.edgemin = event.target.value
			this.vectorizer.tick()
			sessionStorage.setItem("tinqta:edge-min", event.target.value)
		})
		
		this.shadow.getElementById("edge-max").addEventListener("change", (event) => {
			this.vectorizer.edgemax = event.target.value
			this.vectorizer.tick()
			sessionStorage.setItem("tinqta:edge-max", event.target.value)
		})
		
		this.shadow.getElementById("edge-detail").addEventListener("change", (event) => {
			this.vectorizer.edgeDetails = event.target.value
			this.vectorizer.tick()
			sessionStorage.setItem("tinqta:edge-detail", event.target.value)
		})
		
		
		
		
		
	}


	connectedCallback() {
		
		
	}

	saveSVG() {
		// Get the SVG element from the DOM
		
		this.svg.querySelectorAll("path").forEach( p => {
			p.style.strokeDasharray = 'none'
			p.style.strokeDashoffset = 0
		})
		
		// Serialize the SVG element to a string
		const serializer = new XMLSerializer();
		let svgString = serializer.serializeToString(this.svg);
		
		// Add XML declaration
		svgString = '<?xml version="1.0" standalone="no"?>\r\n' + svgString;
		
		// Create a Blob with the SVG string
		const svgBlob = new Blob([svgString], {type: "image/svg+xml;charset=utf-8"});
		
		// Create a download link
		const downloadLink = document.createElement("a");
		downloadLink.href = URL.createObjectURL(svgBlob);
		downloadLink.download = "vector_image.svg";
		
		// Append to body, trigger click, and remove
		document.body.appendChild(downloadLink);
		downloadLink.click();
		document.body.removeChild(downloadLink);

		this.svg.querySelectorAll("path").forEach( p => {
			p.style.strokeDasharray = p.getTotalLength()
			p.style.strokeDashoffset = p.getTotalLength()
		})
	}

	copySVGToClipboard() {


		let animationString = `<style>
			@keyframes dash {
				100% { stroke-dashoffset: 0; }
			}
		</style>`
		
		// Serialize the SVG element to a string
		const serializer = new XMLSerializer();
		let svgString = serializer.serializeToString(this.svg);
		
		navigator.clipboard.writeText(animationString + svgString);

		// Create a popover element for the toast
		const toast = document.createElement('div');
		toast.setAttribute('popover', '');
		toast.id = 'copy-toast';
		toast.textContent = 'SVG copied to clipboard!';
		
		// Style the toast
		toast.style.cssText = `
			background-color: #333;
			color: #fff;
			padding: 10px;
			border-radius: 5px;
			position: fixed;
			bottom: 20px;
			left: 50%;
			transform: translateX(-50%);
			z-index: 1000;
		`;

		// Append the toast to the shadow DOM
		this.shadowRoot.appendChild(toast);

		// Show the toast
		toast.showPopover();

		// Hide the toast after 3 seconds
		setTimeout(() => {
			toast.hidePopover();
		}, 3000);
	}
	
	

}

customElements.define('webcam-generator', WebcamGenerator);
