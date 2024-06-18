'use strict';

export class PaperCanvas extends HTMLElement {
	constructor(n) {
		
		super();
	
		this.shadow = this.attachShadow({ mode: 'open' });

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				
				canvas[resize] {
					width: 100%;
					height: 100%;
				}	
			</style>
			
			<canvas id="paperCanvas" resize="true"></canvas>
				
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));


	}
	


	connectedCallback() {
		paper.install(window)
		let canvas = this.shadow.getElementById('paperCanvas');
		paper.setup(canvas);
		
		let tool = new Tool()
		tool.minDistance = 6

		var path
		this.linelist = []
		this.originalLines = []

		tool.onMouseDown = function (event) {
			path = new Path()
			path.strokeColor = "black"
			path.strokeWidth = 3
			path.strokeCap = 'round'
			path.add(event.point)
		}

		tool.onMouseDrag = function (event) {
			path.add(event.point)
		}

		tool.onMouseUp = () => {
			path.simplify()
			this.processLine(path)
		}
	}
	
	setConfig(config){
		this.config = config
	}
	
	processLine(path) {
		let [segmentedPath, scale, angle] = this.createSegments(path)
		path.scale(scale, path.firstSegment.point)
		path.rotate(angle*360, path.firstSegment.point)
		console.log(scale, angle)

		let points = this.segments2points(segmentedPath)
		this.linelist.push({
			points: points,
			scale: scale,
			rotation: angle,
		})
		this.originalLines.push(path)
	}
	
	createSegments(path) {
		//scale up to normalized size
		let largeDir = Math.max(path.bounds.width, path.bounds.height)
		let baseSize = this.config["stroke_normalizing_size"]
		path.scale(baseSize/largeDir, path.firstSegment.point)
		let scale = largeDir/baseSize
		
		let currAngle = path.lastSegment.point.subtract(
			path.firstSegment.point
		).angle + 180
		
		let angle = currAngle/360
		path.rotate(-currAngle, path.firstSegment.point)
		
		
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
	
	undo(){
		this.linelist.pop()
		let l = this.originalLines.pop()
		l.remove()
	}
	
	drawLine(baseLine, color, smoothing) {
		let points
		if (Array.isArray(baseLine)) {
			points = baseLine
		} else {
			points = baseLine.points
		}

		let path = new Path({segments: points})
		path.strokeColor = color
		path.pivot = path.firstSegment.point
		
		console.log("base", baseLine.position)

		if (!Array.isArray(baseLine)) {
			path.position = new Point(baseLine.position.x * this.config["max_dist"], baseLine.position.y * this.config["max_dist"] )
			console.log("pos", path.position)
			path.scale(baseLine.scale)
			path.rotate(baseLine.rotation * 360)
		} else {
			console.error("no normapization info", baseLine)
		}
		
		if(baseLine.reference_id){
			console.log("moving", originalLines[baseLine.reference_id].firstSegment.point)
			path.translate(originalLines[baseLine.reference_id].firstSegment.point)
		}else{
			path.translate(view.center)
		}
		
		this.processLine(path)

		if (smoothing) {
			path.simplify()
		}
		return path
	}	
	
	processLine(path) {
		let [segmentedPath, scale, angle] = this.createSegments(path)
		path.scale(scale, path.firstSegment.point)
		path.rotate(angle*360, path.firstSegment.point)
		console.log(scale, angle)

		let points = this.segments2points(segmentedPath)
		//let group = drawLine(points, "red")
		//pointlist.push(points)
		this.linelist.push({
			points: points,
			scale: scale,
			rotation: angle,
		})
		this.originalLines.push(path)

	}
	
	createSegments(path) {
		//scale up to normalized size
		let largeDir = Math.max(path.bounds.width, path.bounds.height)
		let baseSize = this.config["stroke_normalizing_size"]
		path.scale(baseSize/largeDir, path.firstSegment.point)
		let scale = largeDir/baseSize
		
		let currAngle = path.lastSegment.point.subtract(
			path.firstSegment.point
		).angle + 180
		
		let angle = currAngle/360
		path.rotate(-currAngle, path.firstSegment.point)
		
		
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
	
	trainingEpoch(data){
		if(!this.trainingLines){
			this.trainingLines = []
			this.animLines = []
		}
		let iteration = []
		for(let [idx,line] of Object.entries(data.lines)){
			let paperline = this.drawLine(line, "grey")
			paperline.strokeWidth = 5
			paperline.opacity = 0.5
			paperline.position = this.originalLines[idx].firstSegment.point
			paperline.scale(this.linelist[idx].scale, paperline.firstSegment.point)
			paperline.rotate(this.linelist[idx].rotation*360, paperline.firstSegment.point)
			paperline.remove()
			
			iteration.push(paperline)
			if(this.animLines.length <= idx){
				let animLine = new Path()
				animLine.strokeColor = "#3DD1E7"
				animLine.strokeWidth = 8
				animLine.opacity = 0.85
				animLine.strokeCap = 'round'
				animLine.strokeJoin = 'round'
				
				this.animLines.push( animLine )
				console.log("push", animLine)
				
			}
			
			if(this.trainingLines[this.trainingLines.length -1]){
				let lastItem = this.trainingLines[this.trainingLines.length -1][idx]
				lastItem.smooth({ type: 'continuous' })
				paperline.smooth({ type: 'continuous' })
				
				this.animLines[idx].tween(1000).onUpdate = (event) => {
					this.animLines[idx].interpolate(lastItem, paperline, event.factor)
				}
				
			}
			
			
		}
		this.trainingLines.push(iteration)
	}

}

customElements.define('paper-canvas', PaperCanvas);
