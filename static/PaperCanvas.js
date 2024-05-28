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

}

customElements.define('paper-canvas', PaperCanvas);
