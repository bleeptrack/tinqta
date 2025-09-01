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
            this.render_points(info.pointlist, info.latent_position_list)
            this.render_lines(info.pointlist, info.latent_position_list)
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
			</style>
			
			<canvas id="renderCanvas" touch-action="none"></canvas>
		`;

	
		this.shadow.appendChild(container.content.cloneNode(true));

		
		
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
        console.log(`Creating sphere ${i} at position:`, pos);
        
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

    render_lines(pointlist, latent_position_list) {
        console.log("render_lines called with:", { pointlist, latent_position_list });
        
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
            
            console.log(`=== Line ${i} ===`);
            console.log("line2D:", line2D);
            console.log("line2D type:", typeof line2D);
            console.log("line2D length:", line2D ? line2D.length : "undefined");
            console.log("latentPos:", latentPos);
            
            // Check if line2D is valid
            if (!line2D || !Array.isArray(line2D)) {
                console.warn(`Line ${i} is not a valid array:`, line2D);
                continue;
            }
            
            // Convert 2D points to 3D points by projecting them onto a plane at the latent position
            const points3D = [];
            
            for (let j = 0; j < line2D.length; j++) {
                const point2D = line2D[j];
                console.log(`Point ${j}:`, point2D, "type:", typeof point2D);
                
                // Check if point2D is valid - it's an object with x,y properties
                if (!point2D || typeof point2D !== 'object' || point2D.x === undefined || point2D.y === undefined) {
                    console.warn(`Point ${j} in line ${i} is not valid:`, point2D);
                    continue;
                }
                
                // Use 2D points directly as 3D points (independent of latent position)
                const x = point2D.x * 0.01; // Scale the 2D coordinates
                const y = point2D.y * 0.01;
                const z = 0; // Keep Z at 0 for flat lines
                
                points3D.push(new Vector3(x, y, z));
                console.log(`Point ${j}: 2D(${point2D.x}, ${point2D.y}) -> 3D(${x}, ${y}, ${z})`);
            }
            
            // Only create line if we have valid points
            if (points3D.length > 0) {
                console.log(`Creating thick 3D line ${i} with ${points3D.length} points`);
                
                // Create a thick line using tube for better visibility
                const line = MeshBuilder.CreateTube(
                    "latentLine" + i,
                    { 
                        path: points3D,
                        radius: 0.01, // Thick line radius
                        tessellation: 8,
                        cap: 2 // Both caps
                    },
                    scene
                );
                
                // Position the line at the latent position
                line.position = new Vector3(latentPos[0], latentPos[1], latentPos[2]);
                
                // Set line color to yellow
                const lineMaterial = new StandardMaterial("lineMat" + i, scene);
                lineMaterial.diffuseColor = new Color3(1, 1, 0); // Yellow
                lineMaterial.emissiveColor = new Color3(0.3, 0.3, 0); // Slight yellow glow
                line.material = lineMaterial;
                
                // Make the tube face the camera (billboard mode Y only)
                line.billboardMode = 2; // BILLBOARDMODE_Y (rotate around Y axis only)
                
                this._latentLines.push(line);
            } else {
                console.warn(`No valid points for line ${i}, skipping`);
            }
        }
        
        console.log("Created", this._latentLines.length, "3D lines total");
    }


	connectedCallback() {
        this.socket.emit("inspect latent", {name: "boxo"});
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
