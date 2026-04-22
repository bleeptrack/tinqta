'use strict';

export class PaperCanvasDraw extends HTMLElement {
	constructor(n) {
		
		super();
	
		this.shadow = this.attachShadow({ mode: 'open' });
		this.saveAnimation = true
		this.recordedData = []
		this.recording = false
		this.linelist = []

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				:host {
					display: block;
					width: 100%;
					height: 100%;
					min-height: 0;
					box-sizing: border-box;
				}
				#paperCanvas {
					display: block;
					width: 100%;
					height: 100%;
				}
			</style>
			
			<canvas id="paperCanvas"></canvas>
			
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));

	}

	clear(){
		paper.project.activeLayer.removeChildren()
	}



	connectedCallback() {
		paper.install(window);
		const canvas = this.shadow.getElementById('paperCanvas');
		paper.setup(canvas);

		this._syncViewSize = () => {
			const w = canvas.clientWidth;
			const h = canvas.clientHeight;
			if (w > 0 && h > 0 && paper.view) {
				paper.view.setViewSize(w, h);
			}
		};

		this._resizeObserver = new ResizeObserver(() => this._syncViewSize());
		this._resizeObserver.observe(canvas);
		requestAnimationFrame(() => this._syncViewSize());
	}

	disconnectedCallback() {
		if (this._resizeObserver) {
			this._resizeObserver.disconnect();
			this._resizeObserver = null;
		}
	}

	
	setConfig(config){
		this.config = config
	}
	


	drawLine(lineJSON, color, smoothing, layer){
		//console.log("DRAWING LINE", lineJSON)
		if(layer){
			layer.activate()
		}

		let path = new Path({segments: lineJSON.points})
		path.strokeColor = color
		path.pivot = path.firstSegment.point

		if(lineJSON.position){
			if(lineJSON.position_type == "absolute"){
				path.position = new Point(lineJSON.position.x , lineJSON.position.y)
			}else{
				console.log("RELATIVE POSITION", lineJSON.position, lineJSON.position_type)
				path.position = new Point(lineJSON.position.x * this.config["max_dist"] , lineJSON.position.y * this.config["max_dist"])
				// * this.config["max_dist"]
			}
		}

		path.scale(lineJSON.scale)
		path.rotate(lineJSON.rotation * 360)

		path.simplify()
		

		if(layer){
			paper.project.layers["lines"].activate()
		}

		return path
	}
	
	processLine(path) {
		let [segmentedPath, scale, angle] = this.createSegments(path)
		path.scale(scale, path.firstSegment.point)
		path.rotate(angle*360, path.firstSegment.point)
		//console.log(scale, angle)

		let points = this.segments2points(segmentedPath)
		return {
			points: points,
			scale: scale,
			rotation: angle,
		}
	}

	createSegments(path) {
		//scale up to normalized size

		let angle = this.calculateAngle(path) / 360
		console.log("angle", angle)
		path.rotate(-angle*360, path.firstSegment.point)
		

		let largeDir = Math.max(path.bounds.width, path.bounds.height)
		let baseSize = this.config["stroke_normalizing_size"]
		path.scale(baseSize/largeDir, path.firstSegment.point)
		let scale = largeDir/baseSize
		
		let segmentedPath = new Path()
		
		let dist = path.length / (this.config.nrPoints - 1)
		
		
		for (let i = 0; i < this.config.nrPoints - 1; i++) {
			let p = path.getPointAt(dist * i).round()
			segmentedPath.addSegment(p)
		}
		segmentedPath.addSegment(path.lastSegment.point.round())

		return [segmentedPath, scale, angle]
	}

	segments2points(path) {
		return path.segments.map((seg) => {
			return {x: seg.point.x, y: seg.point.y}
		})
	}

	calculateAngle(path){
		let info = path.clone()
		info.remove()
		let pca = this.calculatePCA(info.segments.map(seg => seg.point))
		
		
		console.log(pca)
		let line = new Path()
		line.add(new Point(0, 0))
		line.add(new Point(pca.eigenvectors[0].x*200, pca.eigenvectors[0].y*200))
		line.position = pca.center
		line.strokeColor = "red"
		line.strokeWidth = 2
		

		let angle = line.lastSegment.point.subtract(line.firstSegment.point).angle
		//line.rotate(-angle, line.firstSegment.point)
		line.remove()
		return angle
	}

	calculatePCA(points) {
		// Center the data by subtracting means
		const meanX = points.reduce((sum, p) => sum + p.x, 0) / points.length;
		const meanY = points.reduce((sum, p) => sum + p.y, 0) / points.length;
		
		const centeredPoints = points.map(p => ({
			x: p.x - meanX,
			y: p.y - meanY
		}));

		// Calculate covariance matrix
		let xx = 0, xy = 0, yy = 0;
		centeredPoints.forEach(p => {
			xx += p.x * p.x;
			xy += p.x * p.y;
			yy += p.y * p.y;
		});
		xx /= points.length;
		xy /= points.length;
		yy /= points.length;

		// Calculate eigenvalues and eigenvectors
		const trace = xx + yy;
		const det = xx * yy - xy * xy;
		const lambda1 = (trace + Math.sqrt(trace * trace - 4 * det)) / 2;
		const lambda2 = (trace - Math.sqrt(trace * trace - 4 * det)) / 2;
		
		// Calculate principal components (eigenvectors)
		let pc1, pc2;
		if (Math.abs(xy) < 1e-10) {
			pc1 = xx > yy ? {x: 1, y: 0} : {x: 0, y: 1};
			pc2 = xx > yy ? {x: 0, y: 1} : {x: 1, y: 0};
		} else {
			pc1 = {
				x: lambda1 - yy,
				y: xy
			};
			pc2 = {
				x: lambda2 - yy,
				y: xy
			};
			// Normalize vectors
			const mag1 = Math.sqrt(pc1.x * pc1.x + pc1.y * pc1.y);
			const mag2 = Math.sqrt(pc2.x * pc2.x + pc2.y * pc2.y);
			pc1 = {x: pc1.x/mag1, y: pc1.y/mag1};
			pc2 = {x: pc2.x/mag2, y: pc2.y/mag2};
		}

		return {
			eigenvalues: [lambda1, lambda2],
			eigenvectors: [pc1, pc2],
			center: {x: meanX, y: meanY}
		};
	}

	colorByTool(){
		let toolColors = {
			"stamp": "green",
			"visual": "blue",
			"stamp correction": "red",
			"visual correction": "purple",
		}
		for(let line of this.linelist){
			if(line.usedTool){
				line.strokeColor = toolColors[line.usedTool]
			}else{
				line.strokeColor = "black"
			}
		}
	}

}

customElements.define('paper-canvas-draw', PaperCanvasDraw);
