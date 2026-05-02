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
import { ScreenshotTools } from "./node_modules/@babylonjs/core/Misc/screenshotTools.js";
import { Matrix, Viewport } from "./node_modules/@babylonjs/core/Maths/math.js";
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

		this.socket.on("models", (models) => {
			this.updateModelDropdown(models);
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
				.export-row {
					display: flex;
					gap: 8px;
					flex-wrap: wrap;
				}
				.export-row button,
				.export-row select {
					background: #222;
					color: white;
					border: 1px solid #666;
					border-radius: 4px;
					padding: 6px 8px;
					font-size: 12px;
				}
				.export-row button {
					cursor: pointer;
				}
				.iso-view-grid {
					display: grid;
					grid-template-columns: repeat(2, 1fr);
					gap: 6px;
					max-width: 220px;
				}
				.iso-view-grid button {
					background: #222;
					color: white;
					border: 1px solid #666;
					border-radius: 4px;
					padding: 8px 6px;
					font-size: 11px;
					cursor: pointer;
				}
				.iso-view-grid button.active {
					border-color: #7cb342;
					background: #2e3d1f;
				}
			</style>
			
			<div class="controls">
				<div class="control-group">
					<label>Line Model</label>
					<div class="export-row">
						<select id="lineModelSelect"></select>
						<button id="reloadLatent" type="button">Load</button>
					</div>
				</div>
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
						<input type="checkbox" id="showYellowLines">
						<label for="showYellowLines">Show Yellow Lines</label>
					</div>
				</div>
				<div class="control-group">
					<label>Isometric views</label>
					<div class="iso-view-grid">
						<button type="button" class="iso-view-btn" data-iso-index="0" title="Isometric, quadrant 1">Iso 1</button>
						<button type="button" class="iso-view-btn" data-iso-index="1" title="Isometric, quadrant 2">Iso 2</button>
						<button type="button" class="iso-view-btn" data-iso-index="2" title="Isometric, quadrant 3">Iso 3</button>
						<button type="button" class="iso-view-btn" data-iso-index="3" title="Isometric, quadrant 4">Iso 4</button>
					</div>
					<div class="export-row" style="margin-top:8px;">
						<button id="freeOrbitBtn" type="button" title="Resume idle rotation">Free orbit</button>
					</div>
				</div>
				<div class="control-group">
					<label>Other angles</label>
					<div class="export-row">
						<select id="cameraPreset">
							<option value="front">Front</option>
							<option value="side">Side</option>
							<option value="top">Top</option>
						</select>
						<button id="applyCameraPreset" type="button">Apply</button>
						<button id="startViewBtn" type="button" title="View from latent-space center">Start</button>
					</div>
				</div>
				<div class="control-group">
					<label>Export</label>
					<div class="export-row">
						<button id="exportPng" type="button">Export PNG</button>
						<button id="exportSvg" type="button">Export SVG</button>
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
		this.showYellowLines = false;
		this.selectedModelName = "swirls";
		/** @type {AutoRotationBehavior | null} */
		this._autoRotationBehavior = null;
		this._defaultAutoRotationSpeed = 1;
		/** True while one of the four isometric preset buttons is active (auto-rotate off). */
		this._isoViewLocked = false;
		/** Canonical isometric tilt (~54.74° from vertical). */
		this._isoBeta = Math.acos(1 / Math.sqrt(3));
		this._isoRadius = 9;
		this.isometricViews = [
			{ alpha: Math.PI / 4, beta: this._isoBeta, radius: this._isoRadius },
			{ alpha: Math.PI / 4 + Math.PI / 2, beta: this._isoBeta, radius: this._isoRadius },
			{ alpha: Math.PI / 4 + Math.PI, beta: this._isoBeta, radius: this._isoRadius },
			{ alpha: Math.PI / 4 + (3 * Math.PI) / 2, beta: this._isoBeta, radius: this._isoRadius }
		];
		this.cameraPresets = {
			front: { alpha: Math.PI / 2, beta: Math.PI / 2.2, radius: 9 },
			side: { alpha: 0, beta: Math.PI / 2.2, radius: 9 },
			top: { alpha: Math.PI / 2, beta: 0.2, radius: 10 }
		};
		this._axisMeshes = [];
		/** Same scale as tube path in `render_lines` (world units per drawing unit). */
		this.LATENT_LINE_SCALE = 0.005;
		/** Must match axis geometry in `createAxisArrows`. */
		this.axisArrowLength = 1.35;
		this.axisConeH = 0.2;
	}

	requestLatentData() {
		this.socket.emit("inspect latent", { name: this.selectedModelName });
	}

	updateModelDropdown(models) {
		const select = this.shadow.getElementById("lineModelSelect");
		if (!select || !Array.isArray(models)) return;
		const cleanModels = models
			.filter((m) => typeof m === "string" && m.length > 0)
			.sort((a, b) => a.localeCompare(b));
		if (cleanModels.length === 0) return;

		const previousValue = this.selectedModelName;
		select.innerHTML = "";
		for (const modelName of cleanModels) {
			const option = document.createElement("option");
			option.value = modelName;
			option.textContent = modelName;
			select.appendChild(option);
		}

		if (cleanModels.includes(previousValue)) {
			select.value = previousValue;
			this.selectedModelName = previousValue;
		} else {
			select.value = cleanModels[0];
			this.selectedModelName = cleanModels[0];
			this.requestLatentData();
		}
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
                const x = point2D.x * this.LATENT_LINE_SCALE;
                const y = point2D.y * this.LATENT_LINE_SCALE;
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

	/**
	 * X (red), Y (green), Z (blue) arrows at origin for latent space orientation.
	 * @param {Scene} scene
	 */
	createAxisArrows(scene) {
		for (const m of this._axisMeshes) {
			m.dispose();
		}
		this._axisMeshes = [];

		const axisLength = this.axisArrowLength;
		const coneH = this.axisConeH;
		const shaftH = axisLength - coneH;
		const shaftRadius = 0.022;
		const coneBaseD = 0.12;

		const defs = [
			{
				name: "X",
				shaftEnd: new Vector3(shaftH, 0, 0),
				conePos: new Vector3(shaftH + coneH / 2, 0, 0),
				coneRot: new Vector3(0, 0, -Math.PI / 2),
				color: new Color3(0.95, 0.22, 0.22)
			},
			{
				name: "Y",
				shaftEnd: new Vector3(0, shaftH, 0),
				conePos: new Vector3(0, shaftH + coneH / 2, 0),
				coneRot: new Vector3(0, 0, 0),
				color: new Color3(0.28, 0.92, 0.32)
			},
			{
				name: "Z",
				shaftEnd: new Vector3(0, 0, shaftH),
				conePos: new Vector3(0, 0, shaftH + coneH / 2),
				coneRot: new Vector3(Math.PI / 2, 0, 0),
				color: new Color3(0.38, 0.62, 0.98)
			}
		];

		for (const d of defs) {
			const shaft = MeshBuilder.CreateTube(
				`axis${d.name}Shaft`,
				{
					path: [Vector3.Zero(), d.shaftEnd],
					radius: shaftRadius,
					tessellation: 8,
					cap: 2
				},
				scene
			);
			const cone = MeshBuilder.CreateCylinder(
				`axis${d.name}Cone`,
				{
					height: coneH,
					diameterTop: 0,
					diameterBottom: coneBaseD,
					tessellation: 18
				},
				scene
			);
			cone.position.copyFrom(d.conePos);
			cone.rotation.copyFrom(d.coneRot);

			const mat = new StandardMaterial(`axisMat${d.name}`, scene);
			mat.diffuseColor = d.color;
			mat.emissiveColor = new Color3(d.color.r * 0.35, d.color.g * 0.35, d.color.b * 0.35);
			mat.specularColor = Color3.Black();
			shaft.material = mat;
			cone.material = mat;
			shaft.isPickable = false;
			cone.isPickable = false;
			this._axisMeshes.push(shaft, cone);
		}
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
			yellowLinesToggle.checked = this.showYellowLines;
            yellowLinesToggle.addEventListener('change', (e) => {
                this.showYellowLines = e.target.checked;
                // Re-render with new toggle state
                this.updateFilterRanges();
            });
        }

		this.shadow.querySelectorAll(".iso-view-btn").forEach((btn) => {
			btn.addEventListener("click", () => {
				const idx = parseInt(btn.getAttribute("data-iso-index"), 10);
				if (!Number.isNaN(idx)) {
					this.applyIsometricView(idx);
				}
			});
		});

		const freeOrbitBtn = this.shadow.getElementById("freeOrbitBtn");
		if (freeOrbitBtn) {
			freeOrbitBtn.addEventListener("click", () => {
				this.clearIsoViewLock();
			});
		}

		const presetSelect = this.shadow.getElementById("cameraPreset");
		const presetButton = this.shadow.getElementById("applyCameraPreset");
		if (presetButton && presetSelect) {
			presetButton.addEventListener("click", () => {
				this.clearIsoViewLock();
				this.applyCameraPreset(presetSelect.value);
			});
		}
		const startViewBtn = this.shadow.getElementById("startViewBtn");
		if (startViewBtn) {
			startViewBtn.addEventListener("click", () => {
				this.applyCenterLookoutView();
			});
		}

		const exportPngButton = this.shadow.getElementById("exportPng");
		if (exportPngButton) {
			exportPngButton.addEventListener("click", () => {
				this.exportPrintPng();
			});
		}

		const exportSvgButton = this.shadow.getElementById("exportSvg");
		if (exportSvgButton) {
			exportSvgButton.addEventListener("click", () => {
				this.exportPrintSvg();
			});
		}

		const lineModelSelect = this.shadow.getElementById("lineModelSelect");
		const reloadLatent = this.shadow.getElementById("reloadLatent");
		if (lineModelSelect) {
			lineModelSelect.value = this.selectedModelName;
			lineModelSelect.addEventListener("change", (e) => {
				this.selectedModelName = e.target.value;
				this.requestLatentData();
			});
		}
		if (reloadLatent) {
			reloadLatent.addEventListener("click", () => {
				this.requestLatentData();
			});
		}
    }

	setCameraAngles(preset) {
		if (!this._camera || !preset) return;
		this._camera.alpha = preset.alpha;
		this._camera.beta = preset.beta;
		this._camera.radius = preset.radius;
		this._camera.target = Vector3.Zero();
	}

	applyCameraPreset(presetName) {
		const preset = this.cameraPresets[presetName];
		if (!preset) return;
		this.setCameraAngles(preset);
	}

	applyCenterLookoutView() {
		if (!this._camera) return;
		this._isoViewLocked = false;
		this.setAutoRotateEnabled(false);
		this.updateIsoViewButtonStyles(-1);
		// ArcRotateCamera cannot stably sit exactly at its target.
		// Keep a tiny offset while targeting origin so "Start" is centered.
		const epsilon = 0.001;
		this._camera.setTarget(Vector3.Zero());
		this._camera.setPosition(new Vector3(0, 0, -epsilon));
	}

	setAutoRotateEnabled(enabled) {
		if (!this._autoRotationBehavior) return;
		this._autoRotationBehavior.idleRotationSpeed = enabled ? this._defaultAutoRotationSpeed : 0;
	}

	updateIsoViewButtonStyles(activeIndex) {
		this.shadow.querySelectorAll(".iso-view-btn").forEach((btn) => {
			const idx = parseInt(btn.getAttribute("data-iso-index"), 10);
			btn.classList.toggle("active", this._isoViewLocked && idx === activeIndex);
		});
	}

	applyIsometricView(index) {
		if (!this._camera) return;
		const preset = this.isometricViews[index];
		if (!preset) return;
		this._isoViewLocked = true;
		this.setAutoRotateEnabled(false);
		this.setCameraAngles(preset);
		this.updateIsoViewButtonStyles(index);
	}

	clearIsoViewLock() {
		this._isoViewLocked = false;
		this.setAutoRotateEnabled(true);
		this.updateIsoViewButtonStyles(-1);
	}

	getExportableLines() {
		if (!this.originalData) return [];
		const { pointlist, latent_position_list, is_original } = this.originalData;
		const exportable = [];
		for (let i = 0; i < pointlist.length; i++) {
			const latentPos = latent_position_list[i];
			if (!this.isPositionInRange(latentPos)) continue;
			const isOriginalLine = is_original[i];
			if (!isOriginalLine && !this.showYellowLines) continue;
			exportable.push({
				index: i,
				line2D: pointlist[i],
				latentPos,
				isOriginalLine
			});
		}
		return exportable;
	}

	/**
	 * Y-only billboard angle (radians) to match mesh `billboardMode` BILLBOARDMODE_Y:
	 * world offset Ry(θ)*(px,py,0) from latent anchor.
	 * @param {number[]} latentPos [x,y,z]
	 */
	getBillboardYAngleRad(latentPos) {
		if (!this._camera) return 0;
		const cam = this._camera.position;
		const lx = latentPos[0];
		const ly = latentPos[1];
		const lz = latentPos[2];
		let vx = cam.x - lx;
		let vy = cam.y - ly;
		let vz = cam.z - lz;
		const len = Math.sqrt(vx * vx + vy * vy + vz * vz);
		if (len < 1e-9) return 0;
		vx /= len;
		vy /= len;
		vz /= len;
		const xzLen = Math.sqrt(vx * vx + vz * vz);
		if (xzLen < 1e-8) return 0;
		const hx = vx / xzLen;
		const hz = vz / xzLen;
		return Math.atan2(hx, hz);
	}

	/**
	 * World position of a latent stroke sample, same as Babylon tube + billboard Y.
	 * @param {number[]} latentPos
	 * @param {number} px scaled local x (already * LATENT_LINE_SCALE)
	 * @param {number} py scaled local y
	 */
	billboardLatentStrokeWorld(latentPos, px, py) {
		const th = this.getBillboardYAngleRad(latentPos);
		const c = Math.cos(th);
		const s = Math.sin(th);
		const lx = latentPos[0];
		const ly = latentPos[1];
		const lz = latentPos[2];
		return new Vector3(lx + px * c, ly + py, lz - px * s);
	}

	/** World-space axis polylines (origin → shaft → tip), same lengths as `createAxisArrows`. */
	getAxisWorldPolylinesForSvg() {
		const L = this.axisArrowLength;
		const shaftH = this.axisArrowLength - this.axisConeH;
		return [
			{ world: [new Vector3(0, 0, 0), new Vector3(shaftH, 0, 0), new Vector3(L, 0, 0)], color: "#e53935", strokeMul: 1.35 },
			{ world: [new Vector3(0, 0, 0), new Vector3(0, shaftH, 0), new Vector3(0, L, 0)], color: "#43a047", strokeMul: 1.35 },
			{ world: [new Vector3(0, 0, 0), new Vector3(0, 0, shaftH), new Vector3(0, 0, L)], color: "#1e88e5", strokeMul: 1.35 }
		];
	}

	downloadTextFile(filename, text, mimeType) {
		const blob = new Blob([text], { type: mimeType });
		const url = URL.createObjectURL(blob);
		const link = document.createElement("a");
		link.href = url;
		link.download = filename;
		link.rel = "noopener";
		// Programmatic download from shadow DOM: append briefly so the browser applies `download`.
		document.body.appendChild(link);
		link.click();
		link.remove();
		URL.revokeObjectURL(url);
	}

	exportPrintPng() {
		if (!this._scene || !this._camera) return;
		this._scene.render();
		const now = new Date().toISOString().replace(/[:.]/g, "-");
		ScreenshotTools.CreateScreenshotUsingRenderTarget(this._engine, this._camera, {
			width: 3000,
			height: 3000
		}, undefined, undefined, false, `latent-space-${now}.png`);
	}

	exportPrintSvg() {
		if (!this._scene || !this._camera) return;
		const canvas = this.shadow.getElementById("renderCanvas");
		if (!canvas) return;

		// Ensure view/projection matrices match the current frame before projecting.
		this._scene.render();

		const width = Math.max(1, canvas.clientWidth);
		const height = Math.max(1, canvas.clientHeight);
		const world = Matrix.Identity();
		const transform = this._scene.getTransformMatrix();
		const viewport = new Viewport(0, 0, width, height);
		const lines = this.getExportableLines();

		let minX = Number.POSITIVE_INFINITY;
		let minY = Number.POSITIVE_INFINITY;
		let maxX = Number.NEGATIVE_INFINITY;
		let maxY = Number.NEGATIVE_INFINITY;

		const expandBbox = (pt) => {
			minX = Math.min(minX, pt.x);
			minY = Math.min(minY, pt.y);
			maxX = Math.max(maxX, pt.x);
			maxY = Math.max(maxY, pt.y);
		};

		/**
		 * @param {Vector3} worldPoint
		 * @param {boolean} strictClip drop points far outside canvas (latent strokes)
		 */
		const projectToScreen = (worldPoint, strictClip) => {
			const projected = Vector3.Project(worldPoint, world, transform, viewport);
			if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y) || !Number.isFinite(projected.z)) return null;
			if (
				strictClip &&
				(projected.x < -width ||
					projected.x > width * 2 ||
					projected.y < -height ||
					projected.y > height * 2)
			) {
				return null;
			}
			const pt = { x: projected.x, y: projected.y, z: projected.z };
			expandBbox(pt);
			return pt;
		};

		/** @type {{ kind: "axis" | "line"; points: { x: number; y: number; z: number }[]; color: string; strokeMul: number; depth?: number }[]} */
		const svgItems = [];
		const dataDepths = [];

		for (const axis of this.getAxisWorldPolylinesForSvg()) {
			const projectedPoints = [];
			for (const w of axis.world) {
				const p = projectToScreen(w, false);
				if (p) projectedPoints.push(p);
			}
			if (projectedPoints.length > 1) {
				svgItems.push({ kind: "axis", points: projectedPoints, color: axis.color, strokeMul: axis.strokeMul });
			}
		}

		for (const entry of lines) {
			const projectedPoints = [];
			const lineMesh = this._scene.getMeshByName(`latentLine${entry.index}`);
			const lineWorldMatrix = lineMesh ? lineMesh.computeWorldMatrix(true) : null;
			for (const point2D of entry.line2D) {
				if (!point2D || point2D.x === undefined || point2D.y === undefined) continue;
				const px = point2D.x * this.LATENT_LINE_SCALE;
				const py = point2D.y * this.LATENT_LINE_SCALE;
				let worldPoint;
				if (lineWorldMatrix) {
					// Match Babylon's current billboard result exactly by transforming local stroke samples
					// through the actual rendered line mesh world matrix.
					worldPoint = Vector3.TransformCoordinates(new Vector3(px, py, 0), lineWorldMatrix);
				} else {
					// Fallback path if mesh lookup fails.
					worldPoint = this.billboardLatentStrokeWorld(entry.latentPos, px, py);
				}
				const p = projectToScreen(worldPoint, true);
				if (p) projectedPoints.push(p);
			}
			if (projectedPoints.length > 1) {
				const depth = projectedPoints.reduce((acc, p) => acc + p.z, 0) / projectedPoints.length;
				dataDepths.push(depth);
				svgItems.push({
					kind: "line",
					points: projectedPoints,
					color: entry.isOriginalLine ? "#ff2b2b" : "#f3de1f",
					strokeMul: entry.isOriginalLine ? 1.15 : 0.8,
					depth
				});
			}
		}

		if (svgItems.length === 0 || !Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
			return;
		}

		const boxWidth = Math.max(maxX - minX, 1);
		const boxHeight = Math.max(maxY - minY, 1);
		const exportSize = 1200;
		const margin = 60;
		const scale = Math.min((exportSize - 2 * margin) / boxWidth, (exportSize - 2 * margin) / boxHeight);
		const scaledWidth = boxWidth * scale;
		const scaledHeight = boxHeight * scale;
		const offsetX = (exportSize - scaledWidth) / 2;
		const offsetY = (exportSize - scaledHeight) / 2;

		const span = Math.max(boxWidth, boxHeight);
		const baseStroke = Math.max(0.8, span * 0.0022 * scale);
		const minDepth = dataDepths.length ? Math.min(...dataDepths) : 0;
		const maxDepth = dataDepths.length ? Math.max(...dataDepths) : 1;
		const depthRange = Math.max(maxDepth - minDepth, 1e-6);
		const lineThicknessBoost = 1.2;

		let svg = "";
		svg += `<?xml version="1.0" encoding="UTF-8"?>\n`;
		svg += `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${exportSize} ${exportSize}" width="${exportSize}" height="${exportSize}">\n`;
		const renderedItems = svgItems.map((polyline) => {
			const points = polyline.points
				.map((point) => {
					const x = (point.x - minX) * scale + offsetX;
					const y = (point.y - minY) * scale + offsetY;
					return `${x.toFixed(2)},${y.toFixed(2)}`;
				})
				.join(" ");
			let strokeColor = polyline.color;
			let strokeWidth = baseStroke * polyline.strokeMul;
			let grayLevel = null;
			if (polyline.kind === "line" && polyline.depth !== undefined) {
				// Camera-depth shading: near lines are dark, far lines become lighter gray.
				const depth01 = (polyline.depth - minDepth) / depthRange;
				const gray = Math.round(15 + depth01 * 175); // near ~#0f0f0f, far ~#bebebe
				strokeColor = `rgb(${gray},${gray},${gray})`;
				strokeWidth = baseStroke * (1.3 - depth01 * 0.45) * polyline.strokeMul * lineThicknessBoost;
				grayLevel = gray;
			}
			return {
				kind: polyline.kind,
				points,
				strokeColor,
				strokeWidth,
				grayLevel
			};
		});

		// Draw farther/lighter lines first, then darker/near lines so dark ones are visually on top.
		const axisItems = renderedItems.filter((item) => item.kind === "axis");
		const lineItems = renderedItems
			.filter((item) => item.kind === "line")
			.sort((a, b) => (b.grayLevel ?? 0) - (a.grayLevel ?? 0));

		for (const item of [...axisItems, ...lineItems]) {
			svg += `<polyline points="${item.points}" fill="none" stroke="${item.strokeColor}" stroke-width="${item.strokeWidth.toFixed(2)}" stroke-linecap="round" stroke-linejoin="round"/>\n`;
		}
		svg += `</svg>\n`;

		const now = new Date().toISOString().replace(/[:.]/g, "-");
		this.downloadTextFile(`latent-space-${now}.svg`, svg, "image/svg+xml");
	}


	connectedCallback() {
        this.requestLatentData();
		const canvas = this.shadow.getElementById("renderCanvas"); // Get the canvas element
		const engine = new Engine(canvas, true); // Generate the BABYLON 3D engine
		this._engine = engine;

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
		this._camera = camera;
        camera.attachControl(canvas, true);
        
        // Add auto-rotation behavior
        const autoRotationBehavior = new AutoRotationBehavior();
        autoRotationBehavior.idleRotationSpeed = this._defaultAutoRotationSpeed;
        this._autoRotationBehavior = autoRotationBehavior;
        camera.addBehavior(autoRotationBehavior);
        const light = new HemisphericLight("light1", new Vector3(0, 1, 0), scene);
		this.createAxisArrows(scene);
		// Default framing: first isometric angle, but keep idle auto-rotate until user picks an iso preset.
		this.setCameraAngles(this.isometricViews[0]);
		this._isoViewLocked = false;
		this.updateIsoViewButtonStyles(-1);
        return scene;
    }
	


}

customElements.define('latent-inspector', LatentInspector);
