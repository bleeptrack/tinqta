let edgeLines = []
let patternLines = []
let samplePoints = []

let mlStrokes = []
let mlScales = []
let mlRotations = []
let mlName = ""
var socket = io();
var raster
let segmenter

let segmentBG = false
let edgeDetails = 5
let edgemin = 20
let edgemax = 50
//import removeBackground from './node_modules/@imgly/background-removal/dist/browser.mjs';


//import imglyRemoveBackground from "@imgly/background-removal"
//console.log(window.removeBackground )


    paper.install(window);
        window.onload = async function () {

            paper.setup("paperCanvas");
            paper.view.onFrame = tick  //tick

            // lets do some fun
            var video = document.getElementById('webcam');
            let vidw = video.width
            let vidh = video.height
            //var video = document.getE kralementById('plotimg');
            var canvas = document.getElementById('canvas');
            canvas.width = vidw
            canvas.height = vidh
            var ctx = canvas.getContext('2d');


            raster = new Raster([vidw,vidh]);


            let models = [

                    //"jojo",
                    //"sun",
                    //"boxes",
                    //"gigswirl",
                    //"trianglestripe",
                    //"boxgroup"
                    "test5",
                    //"pigtail",

                ]
            mlName = _.sample(models)
            socket.emit('generate', {nr: 200, name:mlName});


            // Move the raster to the center of the view
            //raster.position = view.center;
            //vectorizeRaster(raster)
            //generateImage()

            socket.on('result', function(data) {
                console.log(data)
                mlStrokes = data.list
                mlScales = data.scales
                mlRotations = data.rotations
                console.log(mlStrokes.length)

            });



            view.onMouseDown = function(event) {

                /*
                var canvasBG = document.getElementById('background-canvas');
                canvasBG.width = vidw
                canvasBG.height = vidh
                var ctxBG = canvasBG.getContext('2d');
                ctxBG.drawImage(video, 0, 0, vidw, vidh);

                const segmentation = segmenter.segmentPeople(canvasBG, {multiSegmentation: true, segmentBodyParts: false}).then(()=>{
                        console.log("segmentation")

                        const foregroundColor = {r: 0, g: 0, b: 0, a: 0};
                        const backgroundColor = {r: 0, g: 0, b: 0, a: 255};
                        const backgroundDarkeningMask = bodySegmentation.toBinaryMask(
                            segmentation, foregroundColor, backgroundColor).then(() => {
                                const opacity = 0.7;
                                const maskBlurAmount = 3;
                                const flipHorizontal = false;
                                const canvasBG = document.getElementById('background-canvas');
                                // Draw the mask onto the image on a canvas.  With opacity set to 0.7 and
                                // maskBlurAmount set to 3, this will darken the background and blur the
                                // darkened background's edge.
                                bodySegmentation.drawMask(
                                    canvasBG, video, backgroundDarkeningMask, opacity, maskBlurAmount, flipHorizontal);

                            })



                })
                */

                console.log("start")


                let ctx = canvas.getContext('2d');

                raster.setImageData(ctx.getImageData(0, 0, vidw, vidh), [0,0])
                raster.position = view.center;

                let canvasTMP = document.createElement('canvas');
                var context = canvasTMP.getContext('2d');
                context.canvas.width  = vidw;
                context.canvas.height = vidh;
                context.drawImage(video, 0, 0, vidw, vidh);

                const colorThief = new ColorThief();
                let res = colorThief.getPalette(canvasTMP, 10);
                console.log(res)


                //remove background
                const bodypix = ml5.bodyPix(modelReady);
                
                function modelReady() {
                    // segment the image given
                    console.log("video", video)
                    bodypix.segment(video, gotResults);
                    
                }

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
                paper.view.onFrame = undefined
                
            }


            if (navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(function (stream) {
                    video.srcObject = stream;
                    video.play();

                    })
                    .catch(function (err0r) {
                    console.log("Something went wrong!");
                    });
                }



            var options = function(){
                this.blur_radius = 2;
                this.low_threshold = 20;
                this.high_threshold = 50;
            }

            function colorDistance(c1, c2){
                let c = c2.subtract(c1)
                return Math.abs( Math.sqrt( Math.pow(c.red, 2) + Math.pow(c.blue, 2) + Math.pow(c.green, 2) ) )
            }

            function compressColors(raster, palette, backgroundMask){
                
                let mask = new ImageData(backgroundMask,vidw,vidh);
                raster.setImageData(mask, [0,0])
                console.log(raster.getPixel(0,0))
                
                for(let i = 0; i<raster.size.width; i++){
                    for(let j = 0; j<raster.size.height; j++){
                        let col = raster.getPixel(i,j)
                        if(col.alpha == 0){
                            raster.setPixel(i,j, new Color('white'))
                        }else{
                            let closeColIdx = findClosestColor(col, palette)
                            raster.setPixel(i,j, palette[closeColIdx])
                            if(closeColIdx < 1){
                                samplePoints.push([i,j])
                            }
                        }
                        
                    }
                }
                
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

            function drawPattern(raster){
                let points = []
                let used = []
                let whiteColor = new Color('white')

                for(let i = 0; i<400; i++){
                    let x = _.random(0,639)
                    let y = _.random(0,479)

                    while(raster.getPixel(x,y).equals(whiteColor)){
                        x = _.random(0,639)
                        y = _.random(0,479)
                    }

                    let c = new Path.Circle({
                        center: [x,y],
                        radius: 3,
                        fillColor: 'grey'
                    })
                    let p = new Point(x,y)
                    p.ID = points.length
                    points.push(p)

                }




                let remaining = points.length-used.length
                while(remaining > 10){
                    used = drawStroke(points, 40, used)
                    remaining = points.length-used.length
                    console.log("remaining: ", remaining)
                }

            }


            function vectorizeRaster(raster){
                let viewRadius = 3
                let k = 10
                let maxDist = 3.7
                let whiteColor = new Color('white')
                let points = []
                for(let i = 0; i<vidw; i++){
                    for(let j = 0; j<vidh; j++){
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


                /*for(let testPoint of points){
                    //let testPoint = _.sample(points)
                    let sorted = _.sortBy(points, [function(p) { return testPoint.getDistance(p) }]);

                    let count = 1
                    let lastDist = 0

                    while(lastDist <= maxDist && count< 5){
                        if(testPoint.getDistance(sorted[count]) <= maxDist){

                            sorted[count].connections.push(testPoint.ID)
                            testPoint.connections.push(sorted[count].ID)
                            testPoint.connections = _.uniq(testPoint.connections)
                            sorted[count].connections = _.uniq(sorted[count].connections)
                            let path = new Path.Line(testPoint, sorted[count])
                            path.strokeColor = 'green'
                            path.strokeWidth = 2

                            count++

                        }
                        else{
                            lastDist = testPoint.getDistance(sorted[count])+ 1000
                        }
                    }


                }*/

                /*let start = _.sample(points)
                let path = new Path({
                    strokeColor : 'blue',
                    strokeWidth : 3
                })
                path.add(start)
                let used = [start.ID]
                for(let i = 0; i<200; i++){

                    let next = _.sample(_.without(start.connections, ...used))
                    if(next){
                        console.log(next)
                        path.add(points[next])
                        start = points[next]
                        used.push(start.ID)
                    }else{
                        break
                    }

                }*/





            generateImage(points, maxDist)
            }

            function generateImage(points, maxDist){
                let used = []
                let remaining = points.length-used.length
                while(remaining > 100){
                    used = drawStroke(points, maxDist, used)
                    remaining = points.length-used.length
                    console.log("remaining: ", remaining)
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
                            console.log("break at ", i)
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
                            console.log("break at ", i)
                            break
                        }else{
                            path.add(points[sorted[0]])
                            used.push(sorted[0])
                        }
                    }

                    path.simplify(10)
                    //path.smooth()
                    edgeLines.push(path)

                    return used
            }


            function findStartPoints(points){
                let startIDs = []
                for(let testPoint of points){
                    let sorted = _.sortBy(points, [function(p) { return testPoint.getDistance(p) }]);

                    let dir = sorted[1].subtract(testPoint)
                    let dir2 = sorted[2].subtract(testPoint)

                    let angle = Math.abs(dir.getAngle(dir2)-180)
                    if(angle < 15){
                        testPoint.circle.fillColor = 'yellow'
                    }
                }

            }





            function tick() {





                    if(ctx){
                    ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, vidw, vidh);
                    var imageData = ctx.getImageData(0, 0, vidw, vidh);


                    var img_u8 = new jsfeat.matrix_t(vidw, vidh, jsfeat.U8_t | jsfeat.C1_t);
                    jsfeat.imgproc.grayscale(imageData.data, vidw, vidh, img_u8);

                    //jsfeat.imgproc.equalize_histogram(img_u8, img_u8)


                    var r = edgeDetails; //5 bei zu viel gedöns?
                    var kernel_size = (r+1) << 1;


                    jsfeat.imgproc.gaussian_blur(img_u8, img_u8, kernel_size, 0);



                    jsfeat.imgproc.canny(img_u8, img_u8, edgemin, edgemax);



                    // render result back to canvas
                    var data_u32 = new Uint32Array(imageData.data.buffer);
                    var alpha = (0xff << 24);
                    var i = img_u8.cols*img_u8.rows, pix = 0;
                    while(--i >= 0) {
                        pix = img_u8.data[i];
                        data_u32[i] = alpha | (pix << 16) | (pix << 8) | pix;
                    }


                    ctx.putImageData(imageData, 0, 0);
                    console.log("tick")


                }


            }



            function drawML(){
                let points = mlStrokes.pop()
                let scale = mlScales.pop()
                let rot = mlRotations.pop()
                let c = new Path( {segments: points} )
                c.smooth()
                c.strokeColor = 'black'
                c.scale(scale)
                //c.scale(0.4)
                c.scale(0.35)
                //if(mlName != "trianglestripe"){
                //    c.scale(0.85)
                //}else{
                //    c.scale(1.1)
                //}
                c.rotate(rot)
                c.position = samplePoints.pop()
                while( (doesIntersect(c, patternLines) || doesIntersect(c, edgeLines) ) && samplePoints.length > 0 && mlStrokes.length > 0 ){
                    c.position = samplePoints.pop()
                }
                patternLines.push(c)
                console.log("remaining sample points:", samplePoints.length, mlStrokes.length)
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

function downloadSVG(){
    raster.remove()
    var svg = project.exportSVG({ asString: true, bounds: 'content' });
    var svgBlob = new Blob([svg], {type:"image/svg+xml;charset=utf-8"});
    var svgUrl = URL.createObjectURL(svgBlob);
    var downloadLink = document.createElement("a");
    downloadLink.href = svgUrl;
    downloadLink.download = "stamp.svg";
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}
