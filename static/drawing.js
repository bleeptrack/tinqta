
var config

let linelist = []
let originalLines = []

var socket = io()
socket.on("connect", function () {
    socket.emit("my event", {data: "I'm connected!"})
})

socket.on("init", function (data) {
    console.log(data)
    config = data
})

socket.on("flower-result", function (data) {
    project.activeLayer.removeChildren();
    console.log(data)
    let x = 50
    let y = 50
    let animate = false
    for (let [idx,points] of Object.entries(data)) {
        //let line = drawLine(points, Color.random(), true)
        let line = drawLine(points, "black", true)
        line.strokeWidth = 5
        line.opacity = 0.3
        line.position = view.bounds.center
        if(animate){
            line.dashArray = [line.length, line.length]
            line.dashOffset = line.length*10 + idx*10
            line.tweenTo({dashOffset: 0}, 40000)
        }
        line.rotate(idx*10)
        line.scale(2)
    }
})

socket.on("result", function (data) {
    console.log(data)
    let x = 50
    let y = 50
    for (let [idx, points] of data.list.entries()) {
        //let line = drawLine(points, Color.random(), true)
        let line = drawLine(points, "black", true)
        line.strokeWidth = 5
        line.opacity = 0.8
        line.position = Point.random().multiply(view.bounds.size)
        
        if(data.origins){
            let o = drawLine(data.origins[idx], "red", true)
            o.position = line.position
        }


        if(data.scales){
            line.scale(data.scales[idx])
        }
        if(data.rotations){
            line.rotate(data.rotations[idx] *360 )
        }

        line.dashArray = [line.length, line.length]
        line.dashOffset = line.length
        line.tweenTo({dashOffset: 0}, 400)

    }
})

socket.on("prediction", function (data) {
    project.activeLayer.removeChildren()
    clearArea()
    drawPredictionEnsemble(data)
    console.log("known lines: ", linelist.length)
})

socket.on("extention", function (data) {
    
        drawPredictionEnsemble(data)    
    
})

paper.install(window)
window.onload = function () {
    document.getElementById("load").addEventListener("change", load)

    let trainBtn = document.getElementById("train")
    trainBtn.addEventListener("click", () => {
        socket.emit("new dataset", {
            name: document.getElementById("trainname").value,
            list: linelist,
        })
    })
    let trainPatternBtn = document.getElementById("train-pattern")
    trainPatternBtn.addEventListener("click", () => {
        socket.emit("train pattern", {
            name: document.getElementById("trainname").value,
        })
    })

    let genPatternBtn = document.getElementById("generate-pattern")
    genPatternBtn.addEventListener("click", () => {
        socket.emit("generate pattern", {
            name: document.getElementById("trainname").value,
        })
        console.log("emitting")
    })

    let extPatternBtn = document.getElementById("extend-pattern")
    extPatternBtn.addEventListener("click", () => {

        socket.emit("extend pattern", {
            name: document.getElementById("trainname").value,
            list: linelist,
        })
        console.log("emitting")
    })
    
    let generateBtn = document.getElementById("compare")
    generateBtn.addEventListener("click", () =>
        socket.emit("compare", {
            name: document.getElementById("trainname").value
        })
    )

    let compareBtn = document.getElementById("generate")
    compareBtn.addEventListener("click", () =>
        socket.emit("generate", {
            name: document.getElementById("trainname").value,
            nr: 10,
        })
    )

    let flowerBtn = document.getElementById("flower")
    flowerBtn.addEventListener("click", () =>
        socket.emit("flower", {
            name: document.getElementById("trainname").value,
            nr: 20
        })
    )
    //socket.emit('train');

    paper.setup("paperCanvas")
    let tool = new Tool()
    tool.minDistance = 6

    let rect = new Path.Rectangle([10, 10], [100, 100])
    rect.fillColor = "black"

    var path

    tool.onMouseDown = function (event) {
        path = new Path()
        path.strokeColor = "black"
        path.add(event.point)
    }

    tool.onMouseDrag = function (event) {
        path.add(event.point)
    }

    tool.onMouseUp = function (event) {
        path.simplify()

        //socket.emit('new line', points);

        processLine(path)

        /*for(let i = 0; i<3; i++){
            let copy = path.clone()
            copy.scale(Math.random()*0.5 -0.25 + 1)
            copy.rotate(Math.random()*60 -30)
            points = path2points(path)
            let group = drawLine(points, 'blue')
            pointlist.push(points)
        }*/
    }
}

function save() {
    console.log("save")
    let saveArr = []
    for (let o of originalLines) {
        saveArr.push(o.exportJSON())
    }
    download(JSON.stringify(saveArr), "json.txt", "text/plain")
}

function undo(){
    linelist.pop()
    let l = originalLines.pop()
    l.remove()
}

function load(event) {
    //clearArea()

    console.log("LOAD FILE")
    var reader = new FileReader()
    reader.onload = function (event) {
        let loadLines = JSON.parse(event.target.result)
        console.log(loadLines)
        for (let line of loadLines) {
            let p = new Path()
            p.importJSON(line)
            console.log(p)
            processLine(p)
        }

        let outer = findOuterLines()
        outer.forEach((i) => (originalLines[i].strokeColor = "green"))

        socket.emit("raw data", {
            list: linelist,
            outer_ids: outer,
        })
    }
    reader.readAsText(event.target.files[0])
}

function clearArea() {
    originalLines = []
    pointlist = []
}

function download(content, fileName, contentType) {
    var a = document.createElement("a")
    var file = new Blob([content], {type: contentType})
    a.href = URL.createObjectURL(file)
    a.download = fileName
    a.click()
}

function normalizePointpath(points) {
    let p1 = new Point(points[1])
    let p2 = new Point(points[0])
    let dir = p1.subtract(p2)
}

function processLine(path) {
    let [segmentedPath, scale, angle] = createSegments(path)
    path.scale(scale, path.firstSegment.point)
    path.rotate(angle*360, path.firstSegment.point)
    console.log(scale, angle)

    let points = segments2points(segmentedPath)
    //let group = drawLine(points, "red")
    //pointlist.push(points)
    linelist.push({
        points: points,
        scale: scale,
        rotation: angle,
    })
    originalLines.push(path)
    
    
    
}

function createSegments(path) {
    //scale up to normalized size
    largeDir = Math.max(path.bounds.width, path.bounds.height)
    let baseSize = config["stroke_normalizing_size"]
    path.scale(baseSize/largeDir, path.firstSegment.point)
    let scale = largeDir/baseSize
    
    let currAngle = path.lastSegment.point.subtract(
        path.firstSegment.point
    ).angle + 180
    
    let angle = currAngle/360
    path.rotate(-currAngle, path.firstSegment.point)
    
    
    let segmentedPath = new Path()

    let dist = path.length / (config.nrPoints - 1)
    for (let i = 0; i < config.nrPoints - 1; i++) {
        let p = path.getPointAt(dist * i).round()
        segmentedPath.addSegment(p)
    }
    segmentedPath.addSegment(path.lastSegment.point.round())

    return [segmentedPath, scale, angle]
}

function segments2points(path) {
    return path.segments.map((seg) => {
        return {x: seg.point.x, y: seg.point.y}
    })
}



function drawLine(baseLine, color, smoothing) {
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
        path.position = new Point(baseLine.position.x * config["max_dist"], baseLine.position.y * config["max_dist"] )
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
    
    processLine(path)

    if (smoothing) {
        path.simplify()

        return path
    } else {
        let group = new Group()

        for (let p of points) {
            let c = new Path.Circle([p.x, p.y], 2)
            c.fillColor = color
            group.addChild(c)
        }

        group.addChild(path)
        return group
    }
}

function findOuterLines() {
    let rect = new Path.Rectangle(view.bounds)
    let positions = []

    let ids = []
    for (let i = 0; i < rect.length; i += 10) {
        let pt = rect.getPointAt(i)
        let dist = view.bounds.height * 3
        let found_id = -1
        for (let [idx, line] of originalLines.entries()) {
            if (pt.getDistance(line.position) < dist) {
                dist = pt.getDistance(line.position)
                found_id = idx
            }
        }
        ids.push(found_id)
    }
    let unique = [...new Set(ids)]
    return unique
}

function drawPredictionEnsemble(data) {
    console.log(data)

    
    //pointZero = baseList[0]["points"][0]

    if(data["base_list"]){
        let baseList = data["base_list"]
        for (baseLine of baseList) {
            console.log(baseLine)
            let line = drawLine(baseLine, "black", true)
        }
    }

    if(data.ground_truth){
        let gt = drawLine(data.ground_truth, "blue", true)
    }
    
    if(data.prediction){
        if(Array.isArray(data.prediction)){
            for(let p of data.prediction){
                let l=drawLine(p, "red", true) 
                console.log(l)
            }
        }else{
            drawLine(data.prediction, "red", true)            
        }
    }

    /*
    let gt = drawLine(data.ground_truth.points, 'blue', true)
    gt.pivot = gt.firstSegment.point
    gt.position = Point.random().multiply(view.bounds.size)
    gt.scale(1/data.ground_truth.scale)
    gt.rotation = data.ground_truth.rotation
    gt.strokeWidth = 3


    let line = drawLine(data.prediction.points, 'black', true)
    line.pivot = line.firstSegment.point
    line.position = gt.position
    line.scale(1/data.prediction.scale)
    line.rotation = data.prediction.rotation
    */
}

function downloadSVG(){
    var svg = project.exportSVG({ asString: true, bounds: 'content' });
    var svgBlob = new Blob([svg], {type:"image/svg+xml;charset=utf-8"});
    var svgUrl = URL.createObjectURL(svgBlob);
    var downloadLink = document.createElement("a");
    downloadLink.href = svgUrl;
    downloadLink.download = "export.svg";
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}
