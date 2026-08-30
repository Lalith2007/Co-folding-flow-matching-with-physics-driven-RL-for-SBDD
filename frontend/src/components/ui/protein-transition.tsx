import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

export interface ProteinTransitionProps {
  title?: string;
  subtitle?: string;
  tagline?: string;
}

export default function ProteinTransition({
  title = "STRUCTURAL BIOPHYSICAL SPACE",
  subtitle = "Equivariant SE(3) vector fields continuously deform Gaussian noise distributions into atomically precise, thermodynamically stable pocket-docked small molecules.",
  tagline = "IN-SITU CONFORMATIONAL BRIDGE"
}: ProteinTransitionProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasContainerRef = useRef<HTMLDivElement | null>(null);
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const container = containerRef.current;
    const canvasContainer = canvasContainerRef.current;
    if (!container || !canvasContainer) return;

    // ── 1. Three.js Scene Setup ──
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x020617, 0.045);

    const camera = new THREE.PerspectiveCamera(
      45,
      canvasContainer.clientWidth / canvasContainer.clientHeight,
      0.1,
      100
    );
    camera.position.set(0, 0, 14);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
    canvasContainer.appendChild(renderer.domElement);

    // ── 2. Cinematic Lighting ──
    const ambientLight = new THREE.AmbientLight(0x0f172a, 1.5);
    scene.add(ambientLight);

    const cyanPointLight = new THREE.PointLight(0x06b6d4, 4.5, 25);
    cyanPointLight.position.set(-6, 4, 6);
    scene.add(cyanPointLight);

    const emeraldPointLight = new THREE.PointLight(0x10b981, 5.0, 25);
    emeraldPointLight.position.set(6, -4, 6);
    scene.add(emeraldPointLight);

    const purpleRimLight = new THREE.PointLight(0xa855f7, 3.0, 20);
    purpleRimLight.position.set(0, 8, -4);
    scene.add(purpleRimLight);

    // ── 3. 3D Macromolecular & DNA Helix Assembly ──
    const rootGroup = new THREE.Group();
    scene.add(rootGroup);

    // Materials
    const strandMaterialA = new THREE.MeshPhysicalMaterial({
      color: 0x06b6d4,
      emissive: 0x083344,
      emissiveIntensity: 0.6,
      metalness: 0.75,
      roughness: 0.22,
      clearcoat: 0.8,
      clearcoatRoughness: 0.15,
    });

    const strandMaterialB = new THREE.MeshPhysicalMaterial({
      color: 0x10b981,
      emissive: 0x064e3b,
      emissiveIntensity: 0.6,
      metalness: 0.75,
      roughness: 0.22,
      clearcoat: 0.8,
      clearcoatRoughness: 0.15,
    });

    const basePairMaterial = new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      emissive: 0x334155,
      emissiveIntensity: 0.4,
      metalness: 0.85,
      roughness: 0.3,
    });

    const atomMaterialCyan = new THREE.MeshStandardMaterial({
      color: 0x22d3ee,
      emissive: 0x0891b2,
      emissiveIntensity: 0.8,
      metalness: 0.5,
      roughness: 0.2,
    });

    const atomMaterialEmerald = new THREE.MeshStandardMaterial({
      color: 0x34d399,
      emissive: 0x059669,
      emissiveIntensity: 0.8,
      metalness: 0.5,
      roughness: 0.2,
    });

    // Generate Double Helix & Protein Ribbon Backbone
    const numPoints = 80;
    const helixRadius = 2.4;
    const helixLength = 16.0;
    const turns = 2.5;

    const pointsA: THREE.Vector3[] = [];
    const pointsB: THREE.Vector3[] = [];

    const baseSphereGeo = new THREE.SphereGeometry(0.18, 16, 16);
    const rungCylinderGeo = new THREE.CylinderGeometry(0.065, 0.065, 1, 12);

    for (let i = 0; i <= numPoints; i++) {
      const u = i / numPoints;
      const angle = u * Math.PI * 2 * turns;
      const y = (u - 0.5) * helixLength;
      
      const x1 = Math.cos(angle) * helixRadius;
      const z1 = Math.sin(angle) * helixRadius;
      
      const x2 = Math.cos(angle + Math.PI) * helixRadius;
      const z2 = Math.sin(angle + Math.PI) * helixRadius;

      pointsA.push(new THREE.Vector3(x1, y, z1));
      pointsB.push(new THREE.Vector3(x2, y, z2));

      // Periodic Base Pairs & Atoms
      if (i % 3 === 0 && i > 2 && i < numPoints - 2) {
        // Atom on strand A
        const sphereA = new THREE.Mesh(baseSphereGeo, atomMaterialCyan);
        sphereA.position.set(x1, y, z1);
        rootGroup.add(sphereA);

        // Atom on strand B
        const sphereB = new THREE.Mesh(baseSphereGeo, atomMaterialEmerald);
        sphereB.position.set(x2, y, z2);
        rootGroup.add(sphereB);

        // Connecting Base Pair Rod
        const p1 = new THREE.Vector3(x1, y, z1);
        const p2 = new THREE.Vector3(x2, y, z2);
        const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        const distance = p1.distanceTo(p2);

        const rung = new THREE.Mesh(rungCylinderGeo, basePairMaterial);
        rung.position.copy(midpoint);
        rung.scale.set(1, distance, 1);
        rung.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          new THREE.Vector3().subVectors(p2, p1).normalize()
        );
        rootGroup.add(rung);
      }
    }

    const curveA = new THREE.CatmullRomCurve3(pointsA);
    const curveB = new THREE.CatmullRomCurve3(pointsB);

    const tubeGeoA = new THREE.TubeGeometry(curveA, 120, 0.16, 12, false);
    const tubeGeoB = new THREE.TubeGeometry(curveB, 120, 0.16, 12, false);

    const tubeA = new THREE.Mesh(tubeGeoA, strandMaterialA);
    const tubeB = new THREE.Mesh(tubeGeoB, strandMaterialB);
    rootGroup.add(tubeA);
    rootGroup.add(tubeB);

    // Surrounding Quantum Molecular Dust Particles
    const particleCount = 240;
    const particleGeo = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    const particleColors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const radius = 2.5 + Math.random() * 4.5;
      const theta = Math.random() * Math.PI * 2;
      const py = (Math.random() - 0.5) * 16.0;

      particlePositions[i * 3] = Math.cos(theta) * radius;
      particlePositions[i * 3 + 1] = py;
      particlePositions[i * 3 + 2] = Math.sin(theta) * radius;

      // Color variation between cyan and emerald
      if (Math.random() > 0.5) {
        particleColors[i * 3] = 0.04;     // R
        particleColors[i * 3 + 1] = 0.72; // G
        particleColors[i * 3 + 2] = 0.83; // B
      } else {
        particleColors[i * 3] = 0.06;     // R
        particleColors[i * 3 + 1] = 0.71; // G
        particleColors[i * 3 + 2] = 0.51; // B
      }
    }

    particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    particleGeo.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));

    const particleMat = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending,
    });

    const particles = new THREE.Points(particleGeo, particleMat);
    rootGroup.add(particles);

    // Initial tilt
    rootGroup.rotation.z = THREE.MathUtils.degToRad(32);
    rootGroup.rotation.x = THREE.MathUtils.degToRad(15);

    // ── 4. Interactive Mouse Movement & Parallax ──
    const mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      if (rect.top <= window.innerHeight && rect.bottom >= 0) {
        mouse.targetX = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
        mouse.targetY = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
      }
    };
    window.addEventListener('mousemove', handleMouseMove);

    // ── 5. Scroll State Tracking & Smooth Entrance ──
    const handleScroll = () => {
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const windowHeight = window.innerHeight;
      
      // Calculate intersection progress (0 when entering bottom, 1 when centered, 0 when leaving top)
      const elementCenter = rect.top + rect.height / 2;
      const distanceFromCenter = Math.abs(windowHeight / 2 - elementCenter);
      const maxDistance = windowHeight / 2 + rect.height / 2;
      
      const progress = Math.max(0, Math.min(1, 1 - distanceFromCenter / maxDistance));
      setScrollProgress(progress);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    // ── 6. Resize Handler ──
    const handleResize = () => {
      if (!canvasContainer) return;
      const width = canvasContainer.clientWidth;
      const height = canvasContainer.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // ── 7. Animation Loop ──
    let animationFrameId: number;
    let clock = new THREE.Clock();
    let isVisible = true;

    // IntersectionObserver to pause when out of viewport
    const observer = new IntersectionObserver(
      ([entry]) => {
        isVisible = entry.isIntersecting;
      },
      { threshold: 0.05 }
    );
    observer.observe(container);

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      if (!isVisible) return;

      const elapsedTime = clock.getElapsedTime();

      // Smooth mouse interpolation
      mouse.x += (mouse.targetX - mouse.x) * 0.05;
      mouse.y += (mouse.targetY - mouse.y) * 0.05;

      // Continuous slow majestic rotation + mouse drift
      rootGroup.rotation.y = elapsedTime * 0.22 + mouse.x * 0.45;
      rootGroup.rotation.x = THREE.MathUtils.degToRad(15) + Math.sin(elapsedTime * 0.4) * 0.08 + mouse.y * 0.25;
      
      // Gentle floating breathing motion
      rootGroup.position.y = Math.sin(elapsedTime * 0.7) * 0.35;
      particles.rotation.y = -elapsedTime * 0.08;

      renderer.render(scene, camera);
    };
    animate();

    // ── Cleanup ──
    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleResize);
      observer.disconnect();

      tubeGeoA.dispose();
      tubeGeoB.dispose();
      strandMaterialA.dispose();
      strandMaterialB.dispose();
      basePairMaterial.dispose();
      baseSphereGeo.dispose();
      rungCylinderGeo.dispose();
      atomMaterialCyan.dispose();
      atomMaterialEmerald.dispose();
      particleGeo.dispose();
      particleMat.dispose();
      renderer.dispose();

      if (canvasContainer && canvasContainer.contains(renderer.domElement)) {
        canvasContainer.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <section 
      ref={containerRef}
      className="relative w-full min-h-[580px] lg:min-h-[720px] flex items-center justify-center overflow-hidden bg-slate-950 border-y border-slate-800/60 transition-opacity duration-700"
    >
      {/* Three.js 3D WebGL Canvas Layer */}
      <div 
        ref={canvasContainerRef}
        className="absolute inset-0 w-full h-full pointer-events-none z-10"
      />

      {/* Subtle background ambient glow gradients */}
      <div className="absolute inset-0 bg-radial from-emerald-500/5 via-transparent to-slate-950 pointer-events-none z-0" />
      
      {/* Top & bottom smooth gradient fade into neighboring sections */}
      <div className="absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-slate-950 via-slate-950/70 to-transparent pointer-events-none z-20" />
      <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-slate-950 via-slate-950/80 to-transparent pointer-events-none z-20" />

      {/* Floating Scientific Storytelling Overlay */}
      <div className="relative z-30 max-w-5xl mx-auto px-6 text-center space-y-6 pointer-events-none">
        
        {/* Tagline Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-900/90 border border-slate-700/80 text-cyan-300 text-xs font-mono tracking-widest backdrop-blur-xl shadow-[0_0_30px_rgba(6,182,212,0.25)]">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>{tagline}</span>
        </div>

        {/* Section Headline */}
        <h2 className="text-3xl sm:text-5xl lg:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-white via-slate-100 to-slate-300 tracking-tight leading-tight drop-shadow-[0_8px_30px_rgba(0,0,0,0.9)]">
          {title}
        </h2>

        {/* Narrative Description */}
        <p className="max-w-2xl mx-auto text-sm sm:text-base text-slate-300 font-normal leading-relaxed drop-shadow backdrop-blur-[2px] bg-slate-950/30 p-4 rounded-xl border border-white/5">
          {subtitle}
        </p>

        {/* Micro-Features Floating Pills */}
        <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
          <div className="px-3.5 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300 backdrop-blur-md flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span>Optimal Transport Trajectories</span>
          </div>
          <div className="px-3.5 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300 backdrop-blur-md flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <span>Multi-Objective PPO Co-Folding</span>
          </div>
          <div className="px-3.5 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300 backdrop-blur-md flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
            <span>All-Atom Explicit-Solvent Topology</span>
          </div>
        </div>

      </div>
    </section>
  );
}
