onmessage = function(e) {
	let data = e.data
	
	self.importScripts('https://cdnjs.cloudflare.com/ajax/libs/paper.js/0.12.15/paper-full.min.js')
	self.importScripts('https://cdn.jsdelivr.net/npm/lodash@4.17.21/lodash.min.js')
	
	paper.install(this)
	paper.setup(new Size(1000, 1000));
	
	const offscreen = new OffscreenCanvas(data.vidw, data.vidh);
	var context = offscreen.getContext('2d');
    context.drawImage(data.raster, 0, 0);
	let imageData = context.getImageData(0, 0, data.vidw, data.vidh);
	
	
	this.edgeLines = []
	this.samplePoints = []
	this.patternLines = []
	
	
	findPoints(imageData)
	//vectorizeRaster(data.raster)
	//raster.setImageData(context.getImageData(0, 0, vidw, vidh), [0,0])

	let res = data.res.map( (elem, idx) => (new Color(data.res[idx][0]/255, data.res[idx][1]/255, data.res[idx][2]/255) ) )
	res = res.sort(function(a,b){
		if( a.brightness < b.brightness){
			return -1
		}else{
			return 1
		}
	})

	console.log("SEGMENTATION", data.result)
	compressColors(data.video, res, data.result.partMask)

	//this.shadow.getElementById("webcam").remove()
	//this.shadow.getElementById("edge-canvas").remove()
	
	this.samplePoints = _.shuffle(this.samplePoints)

	
	let allStrokes = data.mlStrokes.length
	let patternLayer = new Layer()
	patternLayer.activate()
	while(this.samplePoints.length > 0 && data.mlStrokes.length > 0){
		drawML()
		
		postMessage({label: "creating shadow", percentage: (allStrokes-data.mlStrokes.length) * 50 / allStrokes + 50});
	}
	
	postMessage({svg: paper.project.exportJSON()});
		
	
	function findPoints(imageData){
		let viewRadius = 3
		let k = 10
		let maxDist = 3.7
		let points = []
		for(let i = 0; i<data.vidw; i++){
			for(let j = 0; j<data.vidh; j++){
				if(isWhitePixel(i,j,imageData)){
					let p = new Point(i,j)
					p.ID = points.length
					points.push(p)
				}
			}
		}

		generateImage(points, maxDist)
	}
	
	function isWhitePixel(x,y, imageData){
		let index = (y*imageData.width + x) * 4;
		let red = imageData.data[index];
		let green = imageData.data[index + 1];
		let blue = imageData.data[index + 2];
		return red == 255 && blue == 255 && green == 255
	}
	
	function setPixel(x,y, imageData, r, g, b){
		let index = (y*imageData.width + x) * 4;
		imageData.data[index] = r
		imageData.data[index + 1] = g
		imageData.data[index + 2] = b
	}
	
	function getPixelAsColor(x,y, imageData){
		let index = (y*imageData.width + x) * 4;
		return new Color(imageData.data[index]/255.0, imageData.data[index + 1]/255.0, imageData.data[index + 2]/255.0)
	}
	
	function isTransparent(x,y, uint8data, width){
		let index = (y*width + x) * 4;
		let alpha = uint8data[index + 3];
		//console.log(uint8data[index + 0], uint8data[index + 1], uint8data[index + 2],uint8data[index + 3] )
		return alpha == 0 || (uint8data[index + 0] == 255 && uint8data[index + 1] == 255 && uint8data[index + 2] == 255)
	}
	
	function generateImage(points, maxDist){
		let used = []
		let remaining = points.length-used.length
		
		while(remaining > 100){
			used = drawStroke(points, maxDist, used)
			remaining = points.length-used.length
			postMessage({label: "detecting lines", percentage: used.length * 50 / points.length});
		}
	}
	
	function drawStroke(points, maxDist, used){
		let ids = points.map(p => p.ID)
		let sampleID = _.sample(_.without(ids, ...used)) //_.sample(points)
		let testPoint = points[sampleID]


			//used = [testPoint.ID]
			used.push(testPoint.ID)
			let path = new Path({
				strokeColor : Color.random(),
				strokeWidth : 1
			})
			path.add(testPoint)


			for(let i = 0; i<500; i++){
				let sorted = _.sortBy(points, [function(p) { return path.lastSegment.point.getDistance(p) }]);
				sorted = sorted.map(p => p.ID)
				sorted = _.without(sorted, ...used)
				if(path.lastSegment.point.getDistance(points[sorted[0]]) > maxDist ){
					break
				}else{
					path.add(points[sorted[0]])
					used.push(sorted[0])
				}
			}
			path.reverse()
			for(let i = 0; i<500; i++){
				let sorted = _.sortBy(points, [function(p) { return path.lastSegment.point.getDistance(p) }]);
				sorted = sorted.map(p => p.ID)
				sorted = _.without(sorted, ...used)
				if(path.lastSegment.point.getDistance(points[sorted[0]]) > maxDist ){
					break
				}else{
					path.add(points[sorted[0]])
					used.push(sorted[0])
				}
			}

			path.simplify(10)
			//path.smooth()
			if(path.length > 0){
				this.edgeLines.push(path)
			}
			

			return used
	}
	
	function findClosestColor(col, palette){
		let dist = 1000
		let idx = 0
		for(let [i, paletteColor] of palette.entries()){
			let d = colorDistance(col, paletteColor)
			
			if(d < dist){
				dist = d
				idx = i
			}
		}
		
		return idx
	}
	
	function colorDistance(c1, c2){
		let c = c2.subtract(c1)
		return Math.abs( Math.sqrt( Math.pow(c.red, 2) + Math.pow(c.blue, 2) + Math.pow(c.green, 2) ) )
	}
	
	function compressColors(imageData, palette, backgroundMask){
		for(let x = 0; x<imageData.width; x++){
			for(let y = 0; y<imageData.height; y++){
				
				if(isTransparent(x, y, backgroundMask, imageData.width)){
					setPixel(x, y, imageData, 255, 255, 255)
				}else{
					let closeColIdx = findClosestColor(getPixelAsColor(x,y,imageData), palette)
					
					setPixel(x, y, imageData, Math.round(palette[closeColIdx].red*255), Math.round(palette[closeColIdx].green*255), Math.round(palette[closeColIdx].blue*255))
					if(closeColIdx < 1){
						this.samplePoints.push([x,y])
					}
				}
			}
		}
	}
	
	function drawML(){
		let points = data.mlStrokes.pop()
		let scale = data.mlScales.pop()
		let rot = data.mlRotations.pop()
		let c = new Path( {segments: points} )
		c.smooth()
		c.strokeColor = 'black'
		
		let largeDir = Math.max(c.bounds.width, c.bounds.height)
		let baseSize = data.baseSize
		c.scale(baseSize/largeDir, c.firstSegment.point)

		//c.scale(0.2)
		//c.scale(scale, c.firstSegment.point)
		c.scale(scale)
		c.scale(0.1)
		c.rotate(rot)
		
		c.position = this.samplePoints.pop()
		while( (doesIntersect(c, this.patternLines) || doesIntersect(c, this.edgeLines) ) && this.samplePoints.length > 0 && data.mlStrokes.length > 0 ){
			c.position = this.samplePoints.pop()
		}
		this.patternLines.push(c)
	}
	
	function doesIntersect(item, arr){
		for(let i of arr){
			if(item.intersects(i)){
				return true
			}
		}
		return false
	}
  
}
