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
			<div class="button-container">
				<button id="downloadJSON">Download JSON</button>
				<button id="downloadSvg">Download SVG</button>
				<input type="file" id="uploadSvg" accept=".json" style="display: none;">
				<label for="uploadSvg" class="upload-button">Upload SVG</label>
			</div>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));
		

		this.shadow.getElementById('downloadSvg').addEventListener('click', () => {
			// Get the SVG from Paper.js project
			const svg = paper.project.exportSVG({ asString: true });

			// Create a Blob with the SVG content
			const blob = new Blob([svg], { type: 'image/svg+xml' });

			// Create a temporary URL for the Blob
			const url = URL.createObjectURL(blob);

			// Create a temporary anchor element
			const downloadLink = document.createElement('a');
			downloadLink.href = url;
			downloadLink.download = 'drawing.svg';

			// Append to body, trigger click, and remove
			document.body.appendChild(downloadLink);
			downloadLink.click();
			document.body.removeChild(downloadLink);

			// Revoke the temporary URL
			URL.revokeObjectURL(url);
		})

		this.shadow.getElementById('downloadJSON').addEventListener('click', () => {
			this.exportLines();
		})

		this.shadow.getElementById('uploadSvg').addEventListener('change', (event) => {
			const file = event.target.files[0];
			if (file) {
				const reader = new FileReader();
				reader.onload = (e) => {
					const contents = e.target.result;
					this.importLines(contents);
				};
				reader.readAsText(file);
			}
		})

	}
	
	setPlaceholder(){
		paper.project.importSVG("/static/placeholder.svg", (svg) => {
			svg.position = view.center
			svg.fitBounds(paper.view.bounds)
			svg.scale(0.9)
			svg.strokeWidth = 2
			this.placeholder = svg
			paper.view.onMouseDown = (event) => {
				this.placeholder.remove()
				paper.view.onMouseDown = null
			}
		})

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
			if(path.segments.length > 1){
				path.simplify()
				this.processLine(path)
			}else{
				path.remove()
			}
		}
	}

	exportLines(){
		// Convert originalLines to JSON
		const jsonData = JSON.stringify(this.originalLines.map(line => line.exportJSON()));

		// Create a Blob with the JSON data
		const blob = new Blob([jsonData], { type: 'application/json' });

		// Create a temporary URL for the Blob
		const url = URL.createObjectURL(blob);

		// Create a temporary anchor element
		const a = document.createElement('a');
		a.href = url;
		a.download = `lines.json`;
		// Append the anchor to the body, click it, and remove it
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);

		// Revoke the temporary URL
		URL.revokeObjectURL(url);
	}

	importLines(data){
		this.originalLines = JSON.parse(data).map(line => {
			let p = new Path()
			p.importJSON(line)
			p.strokeColor = "black"
			p.strokeWidth = 3
			p.strokeCap = 'round'
			return p
		})
		console.log(this.originalLines)
		for(let line of this.originalLines){
			//this.processLine(line)
		}
	}
	
	createMatchingLines(){
		paper.project.layers[0].activate()
		let ids = paper.project.layers[0].children.length
		this.lines2process = [...paper.project.layers[0].children]
		
		this.lines2process.forEach(l => {
			if(l.length > 0){
				this.processLine(l)
			}
		})
		
	}
	
	interpolationData(data, doInterpolation){
		paper.project.layers[0].activate()
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
				if(factor > 0 && doInterpolation){
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
					backup.remove()
				}else{
					group.push(backup)
				}
				segmentedData.remove()
				
			}
			
		}
		paper.project.layers[0].removeChildren()
		paper.project.layers[0].addChildren(group)
		
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

	calculateAngle(path){
		let info = path.clone()
		info.remove()
		let pca = this.calculatePCA(info.segments.map(seg => seg.point))
		//let lr = this.calculateLinearRegression(info.segments.map(seg => seg.point))
		console.log(pca)
		let line = new Path()
		line.add(new Point(0, 0))
		line.add(new Point(pca.eigenvectors[0].x*200, pca.eigenvectors[0].y*200))
		line.position = pca.center
		line.strokeColor = "red"
		line.strokeWidth = 2

		let angle = line.lastSegment.point.subtract(line.firstSegment.point).angle
		//line.rotate(-angle, line.firstSegment.point)

		return angle
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
