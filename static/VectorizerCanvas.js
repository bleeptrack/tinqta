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
	

	activateImage(e){

		this.shadow.getElementById("container").innerHTML = `
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
		console.log(URL.createObjectURL(e.target.files.item(0)))
		removeBackground(URL.createObjectURL(e.target.files.item(0)), bgRemoveConfig).then((blob) => {
		// The result is a blob encoded as PNG. It can be converted to an URL to be used as HTMLImage.src
			const url = URL.createObjectURL(blob);
			
			img.src = url
			console.log(url)
			this.video = img
			
		})
		
		
		
		img.addEventListener("load", ()=>{
			
					
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
	
	activateWebcam(){
		if (navigator.mediaDevices.getUserMedia) {
			navigator.mediaDevices.getUserMedia({ video: true })
				.then((stream) => {
					console.log("stream")
					
					
					this.shadow.getElementById("container").innerHTML = `
						<div id="canvas-container"></div>
						<video id="webcam"></video>
						<canvas id="edge-canvas"></canvas>
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
