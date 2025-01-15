'use strict';
import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js"
import lodash from 'https://cdn.jsdelivr.net/npm/lodash@4.17.21/+esm'
import { removeBackground } from 'https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.5.5/+esm'
import { PaperCanvas } from './PaperCanvas.js'

export class VectorizerCanvas extends HTMLElement {
	constructor(n) {
		
		
		
		super();
		this.socket = io();
		this.shadow = this.attachShadow({ mode: 'open' });
		this.canvas = new PaperCanvas()
		
		this.edgeDetails = 2
		this.edgemin = 20 //20
		this.edgemax = 50  //50
		
		this.edgeLines = []
		this.samplePoints = []
		this.patternLines = []
		
		this.modelName = "random"
		
		
		
		this.socket.on("init", (config) => {
			console.log("config received", config)
			this.config = config
			this.canvas.setConfig(config)
		})
		
		this.socket.on('result', (data) => {
			console.log(data)
			this.mlStrokes = data.list
			this.mlScales = data.scales
			this.mlRotations = data.rotations
			console.log("mlStrokes length", this.mlStrokes.length)
			this.startImageProcess()
		});
		
		this.socket.on("latentLine", (data) => {
			console.log("latentLine received", data)
			console.log(data)
			this.dispatchEvent(new CustomEvent("progress", {detail: {percentage: 90, label: "adjusting edge lines"}}));
			this.canvas.interpolationData(data, false) //0.1
			this.dispatchEvent(new CustomEvent("progress", {detail: {percentage: 100}}))
		})
		

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			
			<link rel="stylesheet" href="/static/style.css">
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				
				
				#container{
					position: relative;
				}
				#webcam{
					position: absolute;
					top: 0px;
					left: 0px;
					
				}
				#edge-canvas{
					position: absolute;
					top: 0px;
					left: 0px;
					opacity: 0.4;
				}
				#canvas-container{
					
				}
				button{
					margin: 10%;
				}
				img{
					position: absolute;
					top: 0px;
					left: 0px;
					
				}
				#start-container{
					display: flex;
				}

				.loader {
					width: 20px;
					height: 12px;
					display: block;
					margin: auto;
					position: relative;
					border-radius: 4px;
					color: #000;
					background: currentColor;
					box-sizing: border-box;
					animation: animloader 0.6s 0.3s ease infinite alternate;
				}
				.loader::after,
				.loader::before {
					content: '';  
					box-sizing: border-box;
					width: 20px;
					height: 12px;
					background: currentColor;
					position: absolute;
					border-radius: 4px;
					top: 0;
					right: 110%;
					animation: animloader  0.6s ease infinite alternate;
				}
				.loader::after {
					left: 110%;
					right: auto;
					animation-delay: 0.6s;
				}

				@keyframes animloader {
					0% {
						width: 20px;
					}
					100% {
						width: 48px;
					}
				}

				#photo-container{
					display: flex;
					flex-direction: row;
					align-items: center;
				}
			</style>
			
			
			<div id="container">
				<div id="start-container">
					<button id="image-chooser" class="scribble">choose photo</button>
					<button id="start-webcam" class="scribble">start webcam</button>
				</div>
			</div>
			
		`;
		
		

	
		this.shadow.appendChild(container.content.cloneNode(true));
		this.shadow.getElementById("start-webcam").addEventListener("click", () => {
			this.activateWebcam()
		})
		this.shadow.getElementById("image-chooser").addEventListener("click", () => {
			let input = document.createElement('input');
			input.type = 'file';
			input.onchange = (e) => {
				this.activateImage(e)
			}
			input.click()
			
		})
		
		

	}
	
	setModelName(name){
		this.modelName = name
		
	}
	
	startProcess(){
		console.log( {nr: 200, name:this.modelName} )
		
		paper.view.onFrame = undefined
		console.log("start")

		let imgData = this.ctx.getImageData(0, 0, this.vidw, this.vidh)
		this.raster.setImageData(imgData, [0,0])
		this.raster.position = view.center;
		//this.shadow.getElementById("webcam").remove()
		this.ctx = undefined

		this.canvasTMP = document.createElement('canvas');
		var context = this.canvasTMP.getContext('2d');
		context.canvas.width  = this.vidw;
		context.canvas.height = this.vidh;
		context.fillStyle = "white"
		context.fillRect(0, 0, this.vidw, this.vidh)
		context.drawImage(this.video, 0, 0, this.vidw, this.vidh);
		
		//let imgData = context.getImageData(0, 0, this.vidw, this.vidh)

		const colorThief = new ColorThief();
		this.res = colorThief.getPalette(this.canvasTMP, 10);
		console.log(this.res)
		
		this.socket.emit('generate', {nr: 200, name:this.modelName})
		
	}
	
	startImageProcess(){
		
	
		console.log("lengths", this.mlStrokes.length )
		
		
		
			
			
			
				const myWorker = new Worker("static/worker-vectorize.js");
				createImageBitmap( this.raster.getImageData() ).then( (bmp) => {
					myWorker.postMessage({
						mlStrokes: this.mlStrokes,
						mlScales: this.mlScales,
						mlRotations: this.mlRotations,
						baseSize: this.config["stroke_normalizing_size"],
						raster: bmp,
						vidw: this.vidw,
						vidh: this.vidh,
						res: this.res,
						video: this.canvasTMP.getContext('2d').getImageData(0,0, this.vidw, this.vidh)
						
					});
				})
				
				myWorker.addEventListener("message", (event) => {
					if(event.data.percentage){
						this.dispatchEvent(new CustomEvent("progress", {detail: event.data}));
					}else if(event.data.svg){
						if(this.shadow.getElementById("webcam")){
							this.shadow.getElementById("webcam").style.visibility = "hidden"
						}
						if(this.shadow.getElementById("image")){
							this.shadow.getElementById("image").style.visibility = "hidden"
						}
						this.shadow.getElementById("edge-canvas").style.visibility = "hidden"
						paper.project.clear()
						paper.project.importJSON(event.data.svg)
						this.addBrandingText()
						

						
						
						
						
						this.canvas.createMatchingLines()
						
						this.socket.emit("convertToLatentspace", {
							name: this.modelName,
							list: this.canvas.linelist,
						})
						
						
						
						
					}else{
						console.log("unknown worker message")
					}
					
				});
				
				
	}
	
	tick(){
		//console.log(this.shadow.getElementById("edge-canvas").getContext('2d'))
		if(this.ctx){
			//this.ctx = canvas.getContext('2d');
			this.ctx.fillStyle = "white"
			this.ctx.fillRect(0, 0, this.vidw, this.vidh)
			this.ctx.drawImage(this.video, 0, 0, this.vidw, this.vidh);
			var imageData = this.ctx.getImageData(0, 0, this.vidw, this.vidh);


			var img_u8 = new jsfeat.matrix_t(this.vidw, this.vidh, jsfeat.U8_t | jsfeat.C1_t);
			jsfeat.imgproc.grayscale(imageData.data, this.vidw, this.vidh, img_u8);

			//jsfeat.imgproc.equalize_histogram(img_u8, img_u8)


			var r = Number(this.edgeDetails) //5 bei zu viel gedöns?
			console.log("edgeDetails", r)
			var kernel_size = (r+1) << 1;
			console.log("kernel_size", kernel_size)


			jsfeat.imgproc.gaussian_blur(img_u8, img_u8, kernel_size, 0);



			jsfeat.imgproc.canny(img_u8, img_u8, this.edgemin, this.edgemax);



			// render result back to canvas
			var data_u32 = new Uint32Array(imageData.data.buffer);
			var alpha = (0xff << 24);
			var i = img_u8.cols*img_u8.rows, pix = 0;
			while(--i >= 0) {
				pix = img_u8.data[i];
				data_u32[i] = alpha | (pix << 16) | (pix << 8) | pix;
			}


			this.ctx.putImageData(imageData, 0, 0);
			console.log("tick")


		}
	}

	addBrandingText(){

		let l3 = new Layer()
		l3.activate()
		l3.name = "refborder"
		project.layers[1].name = "pattern"
		project.layers[0].name = "lines"
		let boundSize = Math.max(project.layers[0].bounds.width, project.layers[0].bounds.height)
		let r = new Rectangle([project.layers[0].bounds.x, project.layers[0].bounds.y], [boundSize, boundSize])
		r.center = project.layers[0].bounds.center

		let rc = new Path.Rectangle(r)
		rc.strokeColor = "red"
		rc.strokeWidth = 3
		rc.scale(1.2)
		
		let point = rc.bounds.bottomLeft
		let width = rc.bounds.width

		project.layers[1].activate()


		let svgText = `<svg width="40.267498mm" height="3.625478mm" viewBox="0 0 40.267498 3.625478" version="1.1" id="svg5" sodipodi:docname="swrtext.svg" inkscape:version="1.2.2 (b0a8486541, 2022-12-01)" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg">
			<sodipodi:namedview id="namedview7" pagecolor="#ffffff" bordercolor="#666666" borderopacity="1.0" inkscape:showpageshadow="2" inkscape:pageopacity="0.0" inkscape:pagecheckerboard="0" inkscape:deskcolor="#d1d1d1" inkscape:document-units="mm" showgrid="false" inkscape:zoom="1.1893044" inkscape:cx="27.326898" inkscape:cy="-49.188416" inkscape:window-width="1920" inkscape:window-height="1170" inkscape:window-x="0" inkscape:window-y="0" inkscape:window-maximized="1" inkscape:current-layer="layer1" />
			<defs id="defs2" />
			<g inkscape:label="Layer 1" inkscape:groupmode="layer" id="layer1" transform="translate(-95.2474,-89.75852)">
				<g inkscape:label="Hershey Text" style="fill:none;stroke:black;stroke-linecap:round;stroke-linejoin:round" id="g508" transform="translate(89.338539,9.4327391)">
				<g transform="translate(5.08,83.82)" id="g506">
					<path d="m 630,567 -63,63 -95,32 H 346 l -94,-32 -63,-63 v -63 l 31,-63 32,-31 63,-32 189,-63 63,-31 31,-32 32,-63 V 94.5 L 567,31.5 472,0 H 346 l -94,31.5 -63,63" style="stroke-width:0.5383in" transform="scale(0.00508,-0.00508)" id="path484" />
					<path d="M 158,662 315,0 M 472,662 315,0 M 472,662 630,0 M 788,662 630,0" style="stroke-width:0.5383in" transform="matrix(0.00508,0,0,-0.00508,3.2004,0)" id="path486" />
					<path d="M 220,662 V 0 m 0,662 h 284 l 94,-32 32,-32 32,-62 V 472 L 630,410 598,378 504,346 H 220 M 441,346 662,0" style="stroke-width:0.5383in" transform="matrix(0.00508,0,0,-0.00508,7.04088,0)" id="path488" />
					<path d="M 189,662 630,0 M 630,662 189,0" style="stroke-width:0.5383in" transform="matrix(0.00508,0,0,-0.00508,12.3241,0)" id="path490" />
					<path d="M 220,662 V 0 m 0,0 h 378" style="stroke-width:0.5383in" transform="matrix(0.00508,0,0,-0.00508,17.4447,0)" id="path492" />
					<path d="M 567,441 V 0 m 0,346 -63,64 -63,31 H 346 L 284,410 220,346 189,252 V 189 L 220,94.5 284,31.5 346,0 h 95 l 63,31.5 63,63" style="stroke-width:0.5383in" transform="matrix(0.00508,0,0,-0.00508,20.1676,0)" id="path494" />
					<path d="M 220,662 V 0 m 0,346 64,64 62,31 h 95 l 63,-31 63,-64 31,-94 V 189 L 567,94.5 504,31.5 441,0 h -95 l -62,31.5 -64,63" style="stroke-width:0.5383in" transform="matrix(0.00508,0,0,-0.00508,23.2054,0)" id="path496" />
					<path d="m 220,504 v 32 l 32,62 32,32 62,32 h 126 l 64,-32 31,-32 31,-62 V 472 L 567,410 504,315 189,0 h 441" style="stroke-width:0.5383in" transform="matrix(0.00508,0,0,-0.00508,28.1635,0)" id="path498" />
					<path d="M 378,662 284,630 220,536 189,378 V 284 L 220,126 284,31.5 378,0 h 63 l 95,31.5 62,94.5 32,158 v 94 l -32,158 -62,94 -95,32 h -63" style="stroke-width:0.5383in" transform="matrix(0.00508,0,0,-0.00508,31.3639,0)" id="path500" />
					<path d="m 220,504 v 32 l 32,62 32,32 62,32 h 126 l 64,-32 31,-32 31,-62 V 472 L 567,410 504,315 189,0 h 441" style="stroke-width:0.5383in" transform="matrix(0.00508,0,0,-0.00508,34.5643,0)" id="path502" />
					<path d="M 567,662 H 252 l -32,-284 32,32 94,31 h 95 l 95,-31 62,-64 32,-94 V 189 L 598,94.5 536,31.5 441,0 H 346 L 252,31.5 220,63 189,126" style="stroke-width:0.5383in" transform="matrix(0.00508,0,0,-0.00508,37.7647,0)" id="path504" />
				</g>
				</g>
			</g>
			</svg>`

		let text = paper.project.importSVG(svgText)
		let targetSize = width/3
		text.scale(targetSize/text.bounds.width)
		text.bounds.bottomLeft = point
		text.translate(new Point(text.bounds.height, -text.bounds.height))

		
	}
	

	activateImage(e){

		this.shadow.getElementById("container").innerHTML = `
			<span class="loader"></span>
			<h3 id="background-remover-title">Please hang on while we remove the background...</h3>
			<div id="canvas-container"></div>
			<img id="image"></img>
			<canvas id="edge-canvas"></canvas>
		`
		let img = this.shadow.getElementById("image")
		
		let bgRemoveConfig = {
			//model: 'isnet' | 'isnet_fp16' | 'isnet_quint8'; // The model to use. (Default "isnet_fp16")
			output: {
				format: 'image/png' //'image/png' | 'image/jpeg' | 'image/webp'; // The output format. (Default "image/png")
				//quality: 0.8; // The quality. (Default: 0.8)
				//type: 'foreground' | 'background' | 'mask'; // The output type. (Default "foreground")
			}
		}

		// Create a new FileReader
		const reader = new FileReader();

		// Set up the FileReader onload function
		reader.onload = (event) => {
			const imgTmp = new Image();
			imgTmp.onload = () => {
				let scaledImg = this.scaleImageTo1080p(imgTmp)
				removeBackground(scaledImg, bgRemoveConfig).then((blob) => {
				// The result is a blob encoded as PNG. It can be converted to an URL to be used as HTMLImage.src
					const url = URL.createObjectURL(blob);
					
					img.src = url
					console.log(url)
					this.video = img
					
				})
			};
			imgTmp.src = event.target.result;
		};
		// Read the first file from the input
		if(e.target){
			reader.readAsDataURL(e.target.files.item(0));
		}else{
			console.log("no file selected", e)
			const imgTmp = new Image();
			imgTmp.onload = () => {
				let scaledImg = this.scaleImageTo1080p(imgTmp)
				removeBackground(scaledImg, bgRemoveConfig).then((blob) => {
				// The result is a blob encoded as PNG. It can be converted to an URL to be used as HTMLImage.src
					const url = URL.createObjectURL(blob);
					
					img.src = url
					console.log(url)
					this.video = img
					
				})
			};
			
			// Create a canvas element to draw the video frame
			const canvas = document.createElement('canvas');
			const context = canvas.getContext('2d');

			// Set canvas dimensions to match the video frame
			canvas.width = this.vidw 
			canvas.height = this.vidh 

			// Draw the current video frame to the canvas
			context.drawImage(this.video, 0, 0, canvas.width, canvas.height);

			// Convert the canvas content to a data URL
			const dataURL = canvas.toDataURL('image/png');

			// Use this data URL as the source for our image
			imgTmp.src = dataURL;
		}

		
		
		
		
		img.addEventListener("load", ()=>{
			this.shadow.getElementById("background-remover-title").remove()
					
			this.vidw = img.naturalWidth
			this.vidh = img.naturalHeight
			img.style.width = `${this.vidw}px`
			img.style.height = `${this.vidh}px`
			
			//console.log(img)
			
			//this.video.width = this.vidw
			//this.video.height= this.vidh
			
			
			this.ctx = this.shadow.getElementById("edge-canvas").getContext('2d')
			
			let canvcont = this.shadow.getElementById('canvas-container');
			canvcont.style.width = `${this.vidw}px`
			canvcont.style.height= `${this.vidh}px`
			
			this.edgeCanvas = this.shadow.getElementById('edge-canvas');
			this.edgeCanvas.width = this.vidw
			this.edgeCanvas.height = this.vidh
			
			
			this.shadow.getElementById("canvas-container").appendChild(this.canvas)
			//paper.view.onFrame = this.tick.bind(this)
			this.raster = new Raster([this.vidw,this.vidh]);
			this.tick()
			this.dispatchEvent(new CustomEvent("ready"));
		})
		
		
	}

	scaleImageTo1080p(img) {
		console.log("img", img)
		const MAX_WIDTH = 1920;
		const MAX_HEIGHT = 1080;
		let width = img.width;
		let height = img.height;
		

		// Calculate the scaling factor
		if (width > MAX_WIDTH || height > MAX_HEIGHT) {
			const widthRatio = MAX_WIDTH / width;
			const heightRatio = MAX_HEIGHT / height;
			const scaleFactor = Math.min(widthRatio, heightRatio);

			width = Math.floor(width * scaleFactor);
			height = Math.floor(height * scaleFactor);
		}
		
			// Create a canvas to draw the scaled image
			const canvas = document.createElement('canvas');
			canvas.width = width;
			canvas.height = height;
			const ctx = canvas.getContext('2d');

			// Draw the image at the new size
			ctx.drawImage(img, 0, 0, width, height);

			// Create and return a new image from the canvas
			const scaledImg = canvas.toDataURL();
			console.log("img scaled", scaledImg)
			return scaledImg;
	}
	
	activateWebcam(){
		if (navigator.mediaDevices.getUserMedia) {
			navigator.mediaDevices.getUserMedia({ video: true })
				.then((stream) => {
					console.log("stream")
					
					
					this.shadow.getElementById("container").innerHTML = `
						<div id="canvas-container"></div>
						<video id="webcam"></video>
						<canvas id="edge-canvas"></canvas>
						<div id="photo-container">
							<button id="stop-webcam" class="scribble">take photo</button>
							<button id="download-photo" class="scribble">save photo for later</button>
						</div>
					`

					this.video = this.shadow.getElementById("webcam")
					this.video.srcObject = stream;
					this.video.play();
					
					this.video.addEventListener("playing", () => {
						this.vidw = this.video.videoWidth
						this.vidh = this.video.videoHeight
						
						
						this.ctx = this.shadow.getElementById("edge-canvas").getContext('2d')
						
						this.video.width = this.vidw
						this.video.height= this.vidh
						let canvcont = this.shadow.getElementById('canvas-container');
						canvcont.style.width = `${this.vidw}px`
						canvcont.style.height= `${this.vidh}px`
						
						this.edgeCanvas = this.shadow.getElementById('edge-canvas');
						this.edgeCanvas.width = this.vidw
						this.edgeCanvas.height = this.vidh
						
						
						this.shadow.getElementById("canvas-container").appendChild(this.canvas)
						paper.view.onFrame = this.tick.bind(this)
						this.raster = new Raster([this.vidw,this.vidh]);
						
						console.log(navigator.mediaDevices.getUserMedia)
					})
					
					this.shadow.getElementById("stop-webcam").addEventListener("click", () => {
						paper.view.onFrame = () => {}
						this.activateImage(this.video)
					})

					this.shadow.getElementById("download-photo").addEventListener("click", () => {
					
						const tempCanvas = document.createElement('canvas');
						tempCanvas.width = this.vidw;
						tempCanvas.height = this.vidh;
						const tempCtx = tempCanvas.getContext('2d');
						
						tempCtx.drawImage(this.video, 0, 0, this.vidw, this.vidh);
						const dataURL = tempCanvas.toDataURL('image/jpeg');
						
						const link = document.createElement('a');
						link.href = dataURL;
						const now = new Date();
						const formattedDate = now.toISOString().replace(/:/g, '-').replace(/\..+/, '');
						link.download = `tinqta_photo_${formattedDate}.jpg`;
						document.body.appendChild(link);
						link.click();
						document.body.removeChild(link);
					})
					
					

				})
				.catch(function (err) {
				console.log("Something went wrong!", err);
				});
			}
	}

	connectedCallback() {
	
	}
	

	





	getSVG(asString=true){
		this.raster.remove()
		return project.exportSVG({ asString: asString, bounds: 'content' })
	}


	

}

customElements.define('vectorizer-canvas', VectorizerCanvas);
