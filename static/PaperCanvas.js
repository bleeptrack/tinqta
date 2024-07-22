'use strict';

export class PaperCanvas extends HTMLElement {
	constructor(n) {
		
		super();
	
		this.shadow = this.attachShadow({ mode: 'open' });
		this.saveAnimation = true

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
	
	createMatchingLines(){
		let ids = paper.project.activeLayer.children.length
		this.lines2process = [...paper.project.activeLayer.children]
		let group = new Group()
		
		this.lines2process.forEach(l => {
			if(l.length > 0){
				this.processLine(l)
			}
		})
		
	}
	
	interpolationData(data){
		let group = []
		for(let [idx, match] of data.list.entries()){
			
			if(this.lines2process[idx].length > 0){
				
			
			
				console.log(idx, match)
				let segs = match.map( elem => new Point(elem.x, elem.y))
				
				let line = new Path({segments: segs})
				line.strokeWidth = 1
				line.strokeColor = "blue"
				line.position = view.center
				
				
				
				let backup = this.lines2process[idx].clone()
				backup.strokeColor = 'red'
				line.strokeWidth = 2
				
				
				
				let [segmentedData, scale, angle] = this.createSegments(this.lines2process[idx]) 
				
				let factor = Math.max( Math.min( 1 - (scale*4), 1), 0)
				if(factor > 0){
					let test = new Path()
					//
					test = segmentedData.clone()
				
					test.strokeColor = factor > 0 ? 'blue' : "red"
					
					test.pivot = test.firstSegment.point
				
				
					test.position = line.firstSegment.point
					
					test.interpolate(test, line, factor)
					
					test.position = segmentedData.firstSegment.point
					
					test.smooth({ type: 'continuous' })
					test.scale(scale)
					test.rotate(angle*360)
					group.push(test)
					//group.push(backup)
					
					line.remove()
				}else{
					group.push(backup)
				}
				
				
			}
			
		}
		paper.project.activeLayer.removeChildren()
		paper.project.activeLayer.addChildren(group)
		
	}
	
	setConfig(config){
		this.config = config
	}
	
	processLine(path) {
		let [segmentedPath, scale, angle] = this.createSegments(path)
		path.scale(scale, path.firstSegment.point)
		path.rotate(angle*360, path.firstSegment.point)
		//console.log(scale, angle)

		let points = this.segments2points(segmentedPath)
		this.linelist.push({
			points: points,
			scale: scale,
			rotation: angle,
		})
		this.originalLines.push(path)
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
	
	saveAnimationEpoch(data){
		paper.view.autoUpdate = false
		
		if(!this.trainingLines){
			this.trainingLines = []
			this.animLines = []
		}
		let iteration = []
		let animationFrames = []
		let canvas = this.shadow.getElementById('paperCanvas')
		
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
			
			//this.trainingLines.push(iteration)
		}
			
		
		for(let i = 0; i<= 1; i+=0.1){
			for(let x = 0; x<this.animLines.length; x++){
				if(this.trainingLines[this.trainingLines.length -1]){
					
					let lastItem = this.trainingLines[this.trainingLines.length -1][x]
					let paperline = iteration[x]
					lastItem.smooth({ type: 'continuous' })
					paperline.smooth({ type: 'continuous' })
						
						
					this.animLines[x].interpolate(lastItem, paperline, i)
					
					console.log(i)
					paper.view.update()
				}
			}
			
			//await this.sleep()
			this.downloadCanvas()
		}
		this.trainingLines.push(iteration)
		
	}
	
	
	downloadCanvas(){
		paper.view.pause()
		paper.view.requestUpdate()
		paper.view.update()
		var link = document.createElement('a');
		link.download = 'tinqta.png';
		link.href = this.shadow.getElementById('paperCanvas').toDataURL()
		link.click();
	}
	
	
	sleep() { 
		return new Promise(r => setTimeout(r));
	}

}

customElements.define('paper-canvas', PaperCanvas);
