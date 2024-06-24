'use strict';
import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js"
import lodash from 'https://cdn.jsdelivr.net/npm/lodash@4.17.21/+esm'
import { PaperCanvas } from './PaperCanvas.js'

export class VectorizerCanvas extends HTMLElement {
	constructor(n) {
		
		super();
		this.socket = io();
		this.shadow = this.attachShadow({ mode: 'open' });
		this.canvas = new PaperCanvas()
		
		this.edgeDetails = 5
		this.edgemin = 20 //20
		this.edgemax = 50  //50
		
		this.edgeLines = []
		this.samplePoints = []
		this.patternLines = []
		
		
		
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
			this.dispatchEvent(new CustomEvent("ready"));
		});
		
		

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
					opacity: 0.5;
				}
				#canvas-container{
					width: 640px;
					height: 480px;
				}
			</style>
			
			<div id="container">
				<div id="canvas-container" width="640" height="480"></div>
				<video id="webcam" width="640" height="480"></video>
				<canvas id="edge-canvas"></canvas>
			</div>
			
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));
		
		this.models = [

                    "jojo",
                    ////////"sun",
                    "boxes",
                    "gigswirl",
                    "trianglestripe",
                    "boxgroup"
                    ////"pigtail",

                ]
		let mlName = lodash.sample(this.models)
		this.socket.emit('generate', {nr: 200, name:mlName});
		
		this.shadow.getElementById("canvas-container").appendChild(this.canvas)
		
		
		

	}
	
	startProcess(){
		paper.view.onFrame = undefined
		console.log("start")

		this.raster.setImageData(this.ctx.getImageData(0, 0, this.vidw, this.vidh), [0,0])
		this.raster.position = view.center;
		//this.shadow.getElementById("webcam").remove()
		this.ctx = undefined

		let canvasTMP = document.createElement('canvas');
		var context = canvasTMP.getContext('2d');
		context.canvas.width  = this.vidw;
		context.canvas.height = this.vidh;
		context.drawImage(this.video, 0, 0, this.vidw, this.vidh);

		const colorThief = new ColorThief();
		let res = colorThief.getPalette(canvasTMP, 10);
		console.log(res)


		
		
		//remove background
		
		
		console.log("lengths", this.mlStrokes.length )
		const bodypix = ml5.bodyPix( () => {
			
			//canvasTMP? kommentar unter vectorize raster rein?
			bodypix.segment(this.video, (error, result) => {
				
				if (error) {
					console.log(error);
					return;
				}
				// log the result
				console.log("MASK", result.backgroundMask);
			
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
						res: res,
						result: result
					});
				})
				
				myWorker.addEventListener("message", (event) => {
					this.dispatchEvent(new CustomEvent("progress", {detail: event.data}));
				});
				
		
			/*
			
				
				
			
				this.vectorizeRaster(this.raster)
				//raster.setImageData(context.getImageData(0, 0, vidw, vidh), [0,0])

				res = res.map( (elem, idx) => (new Color(res[idx][0]/255, res[idx][1]/255, res[idx][2]/255) ) )
				res = res.sort(function(a,b){
					if( a.brightness < b.brightness){
						return -1
					}else{
						return 1
					}
				})

				this.compressColors(this.raster, res, result.backgroundMask)

				this.shadow.getElementById("webcam").remove()
				this.shadow.getElementById("edge-canvas").remove()
				
				this.samplePoints = lodash.shuffle(this.samplePoints)

				console.log("lengths", this.samplePoints.length, this.mlStrokes.length )
				while(this.samplePoints.length > 0 && this.mlStrokes.length > 0){
					this.drawML()
				}
				*/
			});
			
		});
		
		
		
	}
	
	tick(){
		//console.log(this.shadow.getElementById("edge-canvas").getContext('2d'))
		if(this.ctx){
			//this.ctx = canvas.getContext('2d');
			this.ctx.drawImage(this.video, 0, 0, this.vidw, this.vidh);
			var imageData = this.ctx.getImageData(0, 0, this.vidw, this.vidh);


			var img_u8 = new jsfeat.matrix_t(this.vidw, this.vidh, jsfeat.U8_t | jsfeat.C1_t);
			jsfeat.imgproc.grayscale(imageData.data, this.vidw, this.vidh, img_u8);

			//jsfeat.imgproc.equalize_histogram(img_u8, img_u8)


			var r = this.edgeDetails; //5 bei zu viel gedöns?
			var kernel_size = (r+1) << 1;


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
	

	
	


	connectedCallback() {
		
		paper.view.onFrame = this.tick.bind(this)
		this.ctx = this.shadow.getElementById("edge-canvas").getContext('2d')
		this.video = this.shadow.getElementById('webcam');
		this.vidw = this.video.width
		this.vidh = this.video.height
		this.edgeCanvas = this.shadow.getElementById('edge-canvas');
		this.edgeCanvas.width = this.vidw
		this.edgeCanvas.height = this.vidh
		this.raster = new Raster([this.vidw,this.vidh]);
		console.log(navigator.mediaDevices.getUserMedia)
		
		 if (navigator.mediaDevices.getUserMedia) {
			navigator.mediaDevices.getUserMedia({ video: true })
				.then((stream) => {
				console.log("stream")
				this.video.srcObject = stream;
				this.video.play();

				})
				.catch(function (err) {
				console.log("Something went wrong!", err);
				});
			}
	}
	

	





	getSVG(){
		this.raster.remove()
		return project.exportSVG({ asString: true, bounds: 'content' })
	}


	

}

customElements.define('vectorizer-canvas', VectorizerCanvas);
