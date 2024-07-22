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
			this.dispatchEvent(new CustomEvent("ready"));
			
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
			this.canvas.interpolationData(data, 0.1)
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
			</style>
			
			<div id="container">
				<div id="canvas-container"></div>
				<video id="webcam"></video>
				<canvas id="edge-canvas"></canvas>
			</div>
			
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));
		
		/*
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
		*/
		
		
		
		
		

	}
	
	setModelName(name){
		this.modelName = name
		
	}
	
	startProcess(){
		console.log( {nr: 200, name:this.modelName} )
		
		paper.view.onFrame = undefined
		console.log("start")

		this.raster.setImageData(this.ctx.getImageData(0, 0, this.vidw, this.vidh), [0,0])
		this.raster.position = view.center;
		//this.shadow.getElementById("webcam").remove()
		this.ctx = undefined

		this.canvasTMP = document.createElement('canvas');
		var context = this.canvasTMP.getContext('2d');
		context.canvas.width  = this.vidw;
		context.canvas.height = this.vidh;
		context.drawImage(this.video, 0, 0, this.vidw, this.vidh);

		const colorThief = new ColorThief();
		this.res = colorThief.getPalette(this.canvasTMP, 10);
		console.log(this.res)
		
		this.socket.emit('generate', {nr: 200, name:this.modelName})
		
	}
	
	startImageProcess(){
		
	
		console.log("lengths", this.mlStrokes.length )
		const bodypix = ml5.bodyPix( () => {
			
			let options = {
				palette: {
					leftFace: {
					id: 0,
					color: [255, 255, 255],
					},
					rightFace: {
					id: 1,
					color: [255, 255, 255],
					},
					rightUpperLegFront: {
					id: 2,
					color: [100, 81, 196],
					},
					rightLowerLegBack: {
					id: 3,
					color: [92, 91, 206],
					},
					rightUpperLegBack: {
					id: 4,
					color: [84, 101, 214],
					},
					leftLowerLegFront: {
					id: 5,
					color: [75, 113, 221],
					},
					leftUpperLegFront: {
					id: 6,
					color: [66, 125, 224],
					},
					leftUpperLegBack: {
					id: 7,
					color: [56, 138, 226],
					},
					leftLowerLegBack: {
					id: 8,
					color: [48, 150, 224],
					},
					rightFeet: {
					id: 9,
					color: [40, 163, 220],
					},
					rightLowerLegFront: {
					id: 10,
					color: [33, 176, 214],
					},
					leftFeet: {
					id: 11,
					color: [29, 188, 205],
					},
					torsoFront: {
					id: 12,
					color: [26, 199, 194],
					},
					torsoBack: {
					id: 13,
					color: [26, 210, 182],
					},
					rightUpperArmFront: {
					id: 14,
					color: [28, 219, 169],
					},
					rightUpperArmBack: {
					id: 15,
					color: [33, 227, 155],
					},
					rightLowerArmBack: {
					id: 16,
					color: [41, 234, 141],
					},
					leftLowerArmFront: {
					id: 17,
					color: [51, 240, 128],
					},
					leftUpperArmFront: {
					id: 18,
					color: [64, 243, 116],
					},
					leftUpperArmBack: {
					id: 19,
					color: [79, 246, 105],
					},
					leftLowerArmBack: {
					id: 20,
					color: [96, 247, 97],
					},
					rightHand: {
					id: 21,
					color: [255, 255, 255],
					},
					rightLowerArmFront: {
					id: 22,
					color: [134, 245, 88],
					},
					leftHand: {
					id: 23,
					color: [255, 255, 255],
					}
				}
			}
			
			//canvasTMP? kommentar unter vectorize raster rein?
			bodypix.segmentWithParts(this.canvasTMP, (error, result) => {
				
				if (error) {
					console.log(error);
					return;
				}
				// log the result
				console.log("MASK", result);
			
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
						result: result,
						video: this.canvasTMP.getContext('2d').getImageData(0,0, this.vidw, this.vidh)
						
					});
				})
				
				myWorker.addEventListener("message", (event) => {
					if(event.data.percentage){
						this.dispatchEvent(new CustomEvent("progress", {detail: event.data}));
					}else if(event.data.svg){
						this.shadow.getElementById("webcam").style.visibility = "hidden"
						this.shadow.getElementById("edge-canvas").style.visibility = "hidden"
						paper.project.clear()
						paper.project.importJSON(event.data.svg)
						
						
						this.canvas.createMatchingLines()
						
						this.socket.emit("convertToLatentspace", {
							name: this.modelName,
							list: this.canvas.linelist,
						})
						
						
						
						this.dispatchEvent(new CustomEvent("progress", {detail: {percentage: 100}}))
					}else{
						console.log("unknown worker message")
					}
					
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
			}, options);
			
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
		
		
		
		 if (navigator.mediaDevices.getUserMedia) {
			navigator.mediaDevices.getUserMedia({ video: true })
				.then((stream) => {
					console.log("stream")
					this.video = this.shadow.getElementById('webcam');
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
	

	





	getSVG(){
		this.raster.remove()
		return project.exportSVG({ asString: true, bounds: 'content' })
	}


	

}

customElements.define('vectorizer-canvas', VectorizerCanvas);
