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
		this.edgemin = 20
		this.edgemax = 50
		
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
		});
		
		

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			
			<link rel="stylesheet" href="/static/style.css">
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				#canvas-container{
					width: 640px;
					height: 480px;
				}
				
			</style>
			
			<div id="container">
				vectorizer
				<button id="start">START</button>
				<div id="canvas-container" width="640" height="480"></div>
				<canvas id="edge-canvas"></canvas>
				<video id="webcam" width="640" height="480"></video>
			</div>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));
		
		this.models = [

                    "jojo",
                    "sun",
                    "boxes",
                    "gigswirl",
                    "trianglestripe",
                    "boxgroup"
                    //"pigtail",

                ]
		let mlName = lodash.sample(this.models)
		this.socket.emit('generate', {nr: 200, name:mlName});
		
		this.shadow.getElementById("canvas-container").appendChild(this.canvas)
		
		this.shadow.getElementById("start").addEventListener("click", () => {
			this.startProcess()
		})
		

	}
	
	startProcess(){
		paper.view.onFrame = undefined
		console.log("start")

		this.raster.setImageData(this.ctx.getImageData(0, 0, this.vidw, this.vidh), [0,0])
		this.raster.position = view.center;
		this.shadow.getElementById("edge-canvas").remove()
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
			
			bodypix.segment(this.video, (error, result) => {
				
				if (error) {
					console.log(error);
					return;
				}
				// log the result
				console.log("MASK", result.backgroundMask);
			
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

				
				this.samplePoints = lodash.shuffle(this.samplePoints)

				console.log("lengths", this.samplePoints.length, this.mlStrokes.length )
				while(this.samplePoints.length > 0 && this.mlStrokes.length > 0){
					this.drawML()
				}
			});
			
		});

		/*
		function gotResults(error, result) {
			if (error) {
				console.log(error);
				return;
			}
			// log the result
			console.log("MASK", result.backgroundMask);
		
			vectorizeRaster(raster)
			//raster.setImageData(context.getImageData(0, 0, vidw, vidh), [0,0])

			res = res.map( (elem, idx) => (new Color(res[idx][0]/255, res[idx][1]/255, res[idx][2]/255) ) )
			res = res.sort(function(a,b){
				if( a.brightness < b.brightness){
					return -1
				}else{
					return 1
				}
			})

			compressColors(raster, res, result.backgroundMask)

			
			samplePoints = _.shuffle(samplePoints)

			while(samplePoints.length > 0 && mlStrokes.length > 0){
				drawML()
			}
			
		
		}
		*/
		
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
	
	vectorizeRaster(raster){
		let viewRadius = 3
		let k = 10
		let maxDist = 3.7
		let whiteColor = new Color('white')
		let points = []
		for(let i = 0; i<this.vidw; i++){
			for(let j = 0; j<this.vidh; j++){
				let noiseX = 0//Math.round(Math.random())

				let noiseY = 0//i%2//Math.round(Math.random())
				if(raster.getPixel(i+noiseX,j+noiseY).equals(whiteColor)){
					let c = new Path.Circle({
						fillColor: 'red',
						center: new Point(i, j),
						radius: 1
					})
					let p = new Point(i+noiseX,j+noiseY)
					p.circle = c
					c.remove()
					p.ID = points.length
					p.connections = []
					points.push(p)
				}
			}
		}

		this.generateImage(points, maxDist)
	}
	
	generateImage(points, maxDist){
		let used = []
		let remaining = points.length-used.length
		while(remaining > 100){
			used = this.drawStroke(points, maxDist, used)
			remaining = points.length-used.length
			console.log("remaining: ", remaining)
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
	
	drawStroke(points, maxDist, used){
		let ids = points.map(p => p.ID)
		let sampleID = lodash.sample(lodash.without(ids, ...used)) //_.sample(points)
		let testPoint = points[sampleID]


			//used = [testPoint.ID]
			used.push(testPoint.ID)
			let path = new Path({
				strokeColor : Color.random(),
				strokeWidth : 1
			})
			path.add(testPoint)


			for(let i = 0; i<500; i++){
				let sorted = lodash.sortBy(points, [function(p) { return path.lastSegment.point.getDistance(p) }]);
				sorted = sorted.map(p => p.ID)
				sorted = lodash.without(sorted, ...used)
				if(path.lastSegment.point.getDistance(points[sorted[0]]) > maxDist ){
					console.log("break at ", i)
					break
				}else{
					path.add(points[sorted[0]])
					used.push(sorted[0])
				}
			}
			path.reverse()
			for(let i = 0; i<500; i++){
				let sorted = lodash.sortBy(points, [function(p) { return path.lastSegment.point.getDistance(p) }]);
				sorted = sorted.map(p => p.ID)
				sorted = lodash.without(sorted, ...used)
				if(path.lastSegment.point.getDistance(points[sorted[0]]) > maxDist ){
					console.log("break at ", i)
					break
				}else{
					path.add(points[sorted[0]])
					used.push(sorted[0])
				}
			}

			path.simplify(10)
			//path.smooth()
			this.edgeLines.push(path)

			return used
	}
	
	compressColors(raster, palette, backgroundMask){
		let mask = new ImageData(backgroundMask,this.vidw,this.vidh);
		raster.setImageData(mask, [0,0])
		console.log(raster.getPixel(0,0))
		
		for(let i = 0; i<raster.size.width; i++){
			for(let j = 0; j<raster.size.height; j++){
				let col = raster.getPixel(i,j)
				if(col.alpha == 0){
					raster.setPixel(i,j, new Color('white'))
				}else{
					let closeColIdx = this.findClosestColor(col, palette)
					raster.setPixel(i,j, palette[closeColIdx])
					if(closeColIdx < 1){
						this.samplePoints.push([i,j])
					}
				}
				
			}
		}
		
	}
	
	findClosestColor(col, palette){
		let dist = 1000
		let idx = 0
		for(let [i, paletteColor] of palette.entries()){
			let d = this.colorDistance(col, paletteColor)
			if(d < dist){
				dist = d
				idx = i
			}
		}
		return idx
	}
	
	colorDistance(c1, c2){
		let c = c2.subtract(c1)
		return Math.abs( Math.sqrt( Math.pow(c.red, 2) + Math.pow(c.blue, 2) + Math.pow(c.green, 2) ) )
	}
	
	drawML(){
		let points = this.mlStrokes.pop()
		let scale = this.mlScales.pop()
		let rot = this.mlRotations.pop()
		let c = new Path( {segments: points} )
		c.smooth()
		c.strokeColor = 'black'
		
		let largeDir = Math.max(c.bounds.width, c.bounds.height)
		let baseSize = this.config["stroke_normalizing_size"]
		c.scale(baseSize/largeDir, c.firstSegment.point)
		
		
		//c.scale(scale)
		console.log("scale", baseSize/largeDir)
		//c.scale(0.4)
		c.scale(0.2)
		//if(mlName != "trianglestripe"){
		//    c.scale(0.85)
		//}else{
		//    c.scale(1.1)
		//}
		c.rotate(rot)
		c.position = this.samplePoints.pop()
		while( (this.doesIntersect(c, this.patternLines) || this.doesIntersect(c, this.edgeLines) ) && this.samplePoints.length > 0 && this.mlStrokes.length > 0 ){
			c.position = this.samplePoints.pop()
		}
		this.patternLines.push(c)
		console.log("remaining sample points:", this.samplePoints.length, this.mlStrokes.length)
	}

	doesIntersect(item, arr){
		for(let i of arr){
			if(item.intersects(i)){
				return true
			}
		}
		return false
	}

	getSVG(){
		this.raster.remove()
		return project.exportSVG({ asString: true, bounds: 'content' })
	}


	

}

customElements.define('vectorizer-canvas', VectorizerCanvas);
