'use strict';
import { Engine } from "./node_modules/@babylonjs/core/Engines/engine.js";
import { Scene } from "./node_modules/@babylonjs/core/scene.js";
import { ArcRotateCamera } from "./node_modules/@babylonjs/core/Cameras/arcRotateCamera.js";
import { AutoRotationBehavior } from "./node_modules/@babylonjs/core/Behaviors/Cameras/autoRotationBehavior.js";
import { Vector3 } from "./node_modules/@babylonjs/core/Maths/math.js";
import { Color3 } from "./node_modules/@babylonjs/core/Maths/math.color.js";
import { Plane } from "./node_modules/@babylonjs/core/Maths/math.plane.js";
import { HemisphericLight } from "./node_modules/@babylonjs/core/Lights/hemisphericLight.js";
import { MeshBuilder } from "./node_modules/@babylonjs/core/Meshes/meshBuilder.js";
import { StandardMaterial } from "./node_modules/@babylonjs/core/Materials/standardMaterial.js";
import { DynamicTexture } from "./node_modules/@babylonjs/core/Materials/Textures/dynamicTexture.js";
import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js";

export class LatentInspector extends HTMLElement {
	constructor(n) {
		
		super();
		this.socket = io();
		this.shadow = this.attachShadow({ mode: 'open' });
		
		
		
		this.socket.on("latent", (info) => {
			console.log("latent", info)
			console.log("pointlist:", info.pointlist)
			console.log("latent_position_list:", info.latent_position_list)
			console.log("is_original:", info.is_original)
			
			// Store original data for filtering
			this.originalData = info;
			
            this.render_points(info.pointlist, info.latent_position_list)
            this.render_lines(info.pointlist, info.latent_position_list, info.is_original)
		})
		
		

		const container = document.createElement('template');

		// creating the inner HTML of the editable list element
		container.innerHTML = `
			<link rel="stylesheet" href="/static/style.css">
			<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
			<style>
				#renderCanvas {
					width: 100%;
					height: 100%;
					touch-action: none;
				}
				.controls {
					position: absolute;
					top: 10px;
					left: 10px;
					background: rgba(0, 0, 0, 0.8);
					padding: 15px;
					border-radius: 8px;
					color: white;
					font-family: Arial, sans-serif;
					z-index: 1000;
				}
				.control-group {
					margin-bottom: 15px;
				}
				.control-group label {
					display: block;
					margin-bottom: 5px;
					font-size: 12px;
					font-weight: bold;
				}
				.slider-container {
					display: flex;
					align-items: center;
					gap: 10px;
				}
				.slider-container input[type="range"] {
					flex: 1;
					margin: 0 5px;
				}
				.slider-container span {
					font-size: 11px;
					min-width: 30px;
					text-align: center;
				}
				.toggle-container {
					display: flex;
					align-items: center;
					gap: 10px;
				}
				.toggle-container input[type="checkbox"] {
					margin: 0;
				}
			</style>
			
			<div class="controls">
				<div class="control-group">
					<label>X Range</label>
					<div class="slider-container">
						<span id="xMinValue">-6</span>
						<input type="range" id="xMin" min="-6" max="6" step="0.1" value="-6">
						<input type="range" id="xMax" min="-6" max="6" step="0.1" value="6">
						<span id="xMaxValue">6</span>
					</div>
				</div>
				<div class="control-group">
					<label>Y Range</label>
					<div class="slider-container">
						<span id="yMinValue">-6</span>
						<input type="range" id="yMin" min="-6" max="6" step="0.1" value="-6">
						<input type="range" id="yMax" min="-6" max="6" step="0.1" value="6">
						<span id="yMaxValue">6</span>
					</div>
				</div>
				<div class="control-group">
					<label>Z Range</label>
					<div class="slider-container">
						<span id="zMinValue">-6</span>
						<input type="range" id="zMin" min="-6" max="6" step="0.1" value="-6">
						<input type="range" id="zMax" min="-6" max="6" step="0.1" value="6">
						<span id="zMaxValue">6</span>
					</div>
				</div>
				<div class="control-group">
					<div class="toggle-container">
						<input type="checkbox" id="showYellowLines" checked>
						<label for="showYellowLines">Show Yellow Lines</label>
					</div>
				</div>
			</div>
			
			<canvas id="renderCanvas" touch-action="none"></canvas>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));

		// Initialize filter ranges
		this.filterRanges = {
			xMin: -6, xMax: 6,
			yMin: -6, yMax: 6,
			zMin: -6, zMax: 6
		};

		// Store original data for filtering
		this.originalData = null;
		
		// Toggle state for yellow lines
		this.showYellowLines = true;
	}

    render_points(pointlist, latent_position_list) {
    console.log("render_points called with:", { pointlist, latent_position_list });
    
    // Remove previous latent points if any
    if (!this._latentSpheres) {
        this._latentSpheres = [];
    }
    // Remove old spheres from the scene
    if (this._scene && this._latentSpheres.length > 0) {
        for (const sphere of this._latentSpheres) {
            sphere.dispose();
        }
        this._latentSpheres = [];
    }

    // Use the stored scene reference
    const scene = this._scene;
    if (!scene) {
        console.error("Scene not available for rendering points");
        return;
    }

    console.log("Scene available, creating", latent_position_list.length, "spheres");

    // For each latent position, create a small red sphere
    for (let i = 0; i < latent_position_list.length; i++) {
        const pos = latent_position_list[i];
        
        // pos is [x, y, z]
        // Create a sphere mesh
        const sphere = MeshBuilder.CreateSphere(
            "latentPoint" + i,
            { diameter: 0.05 },
            scene
        );
        sphere.position = new Vector3(pos[0], pos[1], pos[2]);
        // Set material to red
        const mat = new StandardMaterial("mat" + i, scene);
        mat.diffuseColor = new Color3(1, 0, 0);
        sphere.material = mat;
        this._latentSpheres.push(sphere);
    }
    
    console.log("Created", this._latentSpheres.length, "spheres total");
    }

    render_lines(pointlist, latent_position_list, is_original) {
        console.log("render_lines called with:", { pointlist, latent_position_list, is_original });
        
        // Remove previous lines if any
        if (!this._latentLines) {
            this._latentLines = [];
        }
        // Remove old lines from the scene
        if (this._scene && this._latentLines.length > 0) {
            for (const line of this._latentLines) {
                line.dispose();
            }
            this._latentLines = [];
        }

        // Use the stored scene reference
        const scene = this._scene;
        if (!scene) {
            console.error("Scene not available for rendering lines");
            return;
        }

        console.log("Scene available, creating", pointlist.length, "3D lines");

        // For each line in pointlist, create actual 3D lines
        for (let i = 0; i < pointlist.length; i++) {
            const line2D = pointlist[i];
            const latentPos = latent_position_list[i];
            
            // Check if this line should be visible based on filter ranges
            if (!this.isPositionInRange(latentPos)) {
                continue; // Skip this line if it's outside the filter range
            }
            
            
            
            // Check if line2D is valid
            if (!line2D || !Array.isArray(line2D)) {
                console.warn(`Line ${i} is not a valid array:`, line2D);
                continue;
            }
            
            // Convert 2D points to 3D points by projecting them onto a plane at the latent position
            const points3D = [];
            
            for (let j = 0; j < line2D.length; j++) {
                const point2D = line2D[j];
                
                
                // Check if point2D is valid - it's an object with x,y properties
                if (!point2D || typeof point2D !== 'object' || point2D.x === undefined || point2D.y === undefined) {
                    
                    continue;
                }
                
                // Use 2D points directly as 3D points (independent of latent position)
                const x = point2D.x * 0.005; // Scale the 2D coordinates
                const y = point2D.y * 0.005;
                const z = 0; // Keep Z at 0 for flat lines
                
                points3D.push(new Vector3(x, y, z));
                
            }
            
            // Only create line if we have valid points
            if (points3D.length > 0) {
                
                
                // Determine line properties based on is_original
                // is_original is an array where each element corresponds to each line
                const isOriginalLine = is_original[i];
                console.log(`Line ${i}: is_original[${i}] = ${isOriginalLine}`);
                
                // Skip yellow lines if toggle is off
                if (!isOriginalLine && !this.showYellowLines) {
                    continue;
                }
                
                const radius = isOriginalLine ? 0.02 : 0.01; // Thicker for original lines
                const color = isOriginalLine ? new Color3(1, 0, 0) : new Color3(1, 1, 0); // Red for original, yellow for others
                const emissiveColor = isOriginalLine ? new Color3(0.5, 0, 0) : new Color3(0.3, 0.3, 0); // Red glow for original
                
                // Create a thick line using tube for better visibility
                const line = MeshBuilder.CreateTube(
                    "latentLine" + i,
                    { 
                        path: points3D,
                        radius: radius, // Thicker for original lines
                        tessellation: 8,
                        cap: 2 // Both caps
                    },
                    scene
                );
                
                // Position the line at the latent position
                line.position = new Vector3(latentPos[0], latentPos[1], latentPos[2]);
                
                // Set line color based on is_original
                const lineMaterial = new StandardMaterial("lineMat" + i, scene);
                lineMaterial.diffuseColor = color; // Red for original, yellow for others
                lineMaterial.emissiveColor = emissiveColor; // Red glow for original, yellow glow for others
                line.material = lineMaterial;
                
                // Make the tube face the camera (billboard mode Y only)
                line.billboardMode = 7; // BILLBOARDMODE_Y (rotate around Y axis only)
                
                this._latentLines.push(line);
            } else {
                console.warn(`No valid points for line ${i}, skipping`);
            }
        }
        
        console.log("Created", this._latentLines.length, "3D lines total");
    }

    isPositionInRange(position) {
        const [x, y, z] = position;
        return x >= this.filterRanges.xMin && x <= this.filterRanges.xMax &&
               y >= this.filterRanges.yMin && y <= this.filterRanges.yMax &&
               z >= this.filterRanges.zMin && z <= this.filterRanges.zMax;
    }

    updateFilterRanges() {
        if (!this.originalData) return;
        
        // Re-render with filtered data
        this.render_points(this.originalData.pointlist, this.originalData.latent_position_list);
        this.render_lines(this.originalData.pointlist, this.originalData.latent_position_list, this.originalData.is_original);
    }

    setupSliderControls() {
        const sliders = ['xMin', 'xMax', 'yMin', 'yMax', 'zMin', 'zMax'];
        
        sliders.forEach(sliderId => {
            const slider = this.shadow.getElementById(sliderId);
            const valueSpan = this.shadow.getElementById(sliderId + 'Value');
            
            if (slider && valueSpan) {
                // Update display value
                valueSpan.textContent = slider.value;
                
                // Add event listener
                slider.addEventListener('input', (e) => {
                    valueSpan.textContent = e.target.value;
                    
                    // Update filter ranges
                    const axis = sliderId.charAt(0);
                    const type = sliderId.substring(1);
                    this.filterRanges[sliderId] = parseFloat(e.target.value);
                    
                    // Re-render with new filter
                    this.updateFilterRanges();
                });
            }
        });
        
        // Setup toggle control for yellow lines
        const yellowLinesToggle = this.shadow.getElementById('showYellowLines');
        if (yellowLinesToggle) {
            yellowLinesToggle.addEventListener('change', (e) => {
                this.showYellowLines = e.target.checked;
                // Re-render with new toggle state
                this.updateFilterRanges();
            });
        }
    }


	connectedCallback() {
        this.socket.emit("inspect latent", {name: "grid"});
		const canvas = this.shadow.getElementById("renderCanvas"); // Get the canvas element
		const engine = new Engine(canvas, true); // Generate the BABYLON 3D engine

		// Add your code here matching the playground format

		const scene = this.createScene(engine, canvas); //Call the createScene function
		this._scene = scene; // Store scene reference for later use

		// Register a render loop to repeatedly render the scene
		engine.runRenderLoop(function () {
			scene.render();
		});

		// Watch for browser/canvas resize events
		window.addEventListener("resize", function () {
			engine.resize();
		});

		// Setup slider controls
		this.setupSliderControls();
	}

    createScene(engine, canvas) {
        const scene = new Scene(engine);
        const camera = new ArcRotateCamera("camera1", Math.PI / 2, Math.PI / 4, 4, Vector3.Zero(), scene);
        camera.attachControl(canvas, true);
        
        // Add auto-rotation behavior
        const autoRotationBehavior = new AutoRotationBehavior();
        autoRotationBehavior.idleRotationSpeed = 1;
        camera.addBehavior(autoRotationBehavior);
        const light = new HemisphericLight("light1", new Vector3(0, 1, 0), scene);
        return scene;
    }
	


}

customElements.define('latent-inspector', LatentInspector);
