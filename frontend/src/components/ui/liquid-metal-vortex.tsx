import React, { useRef, useEffect } from "react";

// --- Enhanced Bioluminescent Liquid Metal WebGL Shader ---
const useWebGLShader = (
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  props: { hue: number; complexity: number; speed: number }
) => {
  const webglState = useRef<any>(null);
  const mousePos = useRef({ x: 0.5, y: 0.5, targetX: 0.5, targetY: 0.5 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Get WebGL context
    const gl = canvas.getContext("webgl", { 
      alpha: true, 
      antialias: true,
      powerPreference: "high-performance" 
    });

    if (!gl) {
      console.warn("WebGL not supported, rendering canvas fallback.");
      return;
    }

    const vertexShaderSource = `
      attribute vec2 position;
      varying vec2 vUv;
      void main() {
        vUv = (position + 1.0) * 0.5;
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;

    // Vibrant, Iridescent Liquid Metal & Bioluminescent Fluid Shader
    const fragmentShaderSource = `
      precision highp float;
      varying vec2 vUv;
      uniform float iTime;
      uniform vec2 iResolution;
      uniform vec2 iMouse;
      uniform float uHue;
      uniform float uComplexity;
      uniform float uSpeed;

      // Color transformation
      vec3 hsv2rgb(vec3 c) {
        vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
        return c.z * mix(vec3(1.0), rgb, c.y);
      }

      // 2D Rotation
      mat2 rot(float a) {
        float s = sin(a), c = cos(a);
        return mat2(c, -s, s, c);
      }

      // Smooth noise function
      float hash(vec2 p) {
        p = 50.0 * fract(p * 0.3183099 + vec2(0.71, 0.113));
        return -1.0 + 2.0 * fract(p.x * p.y * (p.x + p.y));
      }

      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
                   mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
      }

      // Fractional Brownian Motion for liquid chrome waves
      float fbm(vec2 p) {
        float v = 0.0;
        float a = 0.5;
        vec2 shift = vec2(100.0);
        mat2 rot2 = rot(0.5);
        for (int i = 0; i < 5; ++i) {
          v += a * noise(p);
          p = rot2 * p * 2.0 + shift;
          a *= 0.5;
        }
        return v;
      }

      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y);
        
        // Fluid time scale
        float t = iTime * 0.35 * uSpeed;
        
        // Interactive mouse distortion
        vec2 mouse = (iMouse - 0.5) * 2.0;
        float distToMouse = length(uv - mouse * 0.5);
        vec2 mouseForce = normalize(uv - mouse * 0.5 + 0.001) / (distToMouse * 4.0 + 1.0);
        
        // Domain warping for liquid chrome flow
        vec2 q = vec2(0.0);
        q.x = fbm(uv + vec2(0.0, t * 0.2) + mouseForce * 0.15);
        q.y = fbm(uv + vec2(1.0, t * 0.15));

        vec2 r = vec2(0.0);
        r.x = fbm(uv + 1.0 * q + vec2(1.7, 9.2) + 0.15 * t);
        r.y = fbm(uv + 1.0 * q + vec2(8.3, 2.8) + 0.126 * t);

        float f = fbm(uv + r * uComplexity * 1.5);

        // Compute simulated normal for 3D metallic highlights
        vec2 eps = vec2(0.01, 0.0);
        float f_x = fbm(uv + eps.xy + r * 1.5) - fbm(uv - eps.xy + r * 1.5);
        float f_y = fbm(uv + eps.yx + r * 1.5) - fbm(uv - eps.yx + r * 1.5);
        vec3 normal = normalize(vec3(-f_x * 5.0, -f_y * 5.0, 1.0));

        // Chrome lighting
        vec3 lightDir = normalize(vec3(0.577, 0.577, 0.577));
        float diff = max(dot(normal, lightDir), 0.0);
        
        // Specular sheen
        vec3 viewDir = vec3(0.0, 0.0, 1.0);
        vec3 halfDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);
        float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);

        // Bioluminescent palette based on uHue (emerald / cyan / deep biotech blue)
        vec3 baseColor1 = hsv2rgb(vec3(uHue / 360.0, 0.85, 0.95));
        vec3 baseColor2 = hsv2rgb(vec3(mod(uHue + 45.0, 360.0) / 360.0, 0.70, 0.85));
        vec3 chromeColor = vec3(0.92, 0.96, 1.0);

        // Color blending
        vec3 color = mix(baseColor1, baseColor2, clamp(f * f * 3.5, 0.0, 1.0));
        color = mix(color, chromeColor, clamp(length(q) * 0.8, 0.0, 1.0));
        
        // Add lighting, metallic gloss and specular sheen
        color += vec3(0.8, 1.0, 0.9) * spec * 1.8;
        color += chromeColor * fresnel * 0.9;
        color *= (0.35 + 0.65 * diff);

        // Subtle dark vignette at edges
        float vignette = 1.0 - smoothstep(0.5, 1.5, length(uv));
        color *= vignette;

        // Output with vibrant alpha
        gl_FragColor = vec4(color, 0.88);
      }
    `;

    const compileShader = (source: string, type: number) => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("Shader compile error: " + gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vertexShader = compileShader(vertexShaderSource, gl.VERTEX_SHADER);
    const fragShader = compileShader(fragmentShaderSource, gl.FRAGMENT_SHADER);
    if (!vertexShader || !fragShader) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program link error: " + gl.getProgramInfoLog(program));
      return;
    }
    gl.useProgram(program);

    const vertices = new Float32Array([
      -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1,
    ]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    const uniformLocations = {
      iTime: gl.getUniformLocation(program, "iTime"),
      iResolution: gl.getUniformLocation(program, "iResolution"),
      iMouse: gl.getUniformLocation(program, "iMouse"),
      uHue: gl.getUniformLocation(program, "uHue"),
      uComplexity: gl.getUniformLocation(program, "uComplexity"),
      uSpeed: gl.getUniformLocation(program, "uSpeed"),
    };

    webglState.current = { gl, program, uniformLocations, vertexBuffer };

    return () => {
      if (gl && !gl.isContextLost()) {
        gl.deleteProgram(program);
        gl.deleteShader(vertexShader);
        gl.deleteShader(fragShader);
        gl.deleteBuffer(vertexBuffer);
      }
    };
  }, [canvasRef]);

  // Mouse move handler
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      mousePos.current.targetX = (e.clientX - rect.left) / rect.width;
      mousePos.current.targetY = 1.0 - (e.clientY - rect.top) / rect.height;
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [canvasRef]);

  // Animation render loop
  useEffect(() => {
    if (!webglState.current) return;

    const { gl, uniformLocations } = webglState.current;
    const startTime = performance.now();
    let animationFrameId: number;

    const handleResize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const displayWidth = Math.floor(canvas.clientWidth * dpr);
      const displayHeight = Math.floor(canvas.clientHeight * dpr);

      if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
        canvas.width = displayWidth;
        canvas.height = displayHeight;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uniformLocations.iResolution, canvas.width, canvas.height);
    };

    window.addEventListener("resize", handleResize);
    handleResize();

    const render = () => {
      const time = (performance.now() - startTime) / 1000.0;

      // Smooth mouse interpolation
      mousePos.current.x += (mousePos.current.targetX - mousePos.current.x) * 0.08;
      mousePos.current.y += (mousePos.current.targetY - mousePos.current.y) * 0.08;

      gl.uniform1f(uniformLocations.iTime, time);
      gl.uniform2f(uniformLocations.iMouse, mousePos.current.x, mousePos.current.y);
      gl.uniform1f(uniformLocations.uHue, props.hue || 165);
      gl.uniform1f(uniformLocations.uComplexity, props.complexity || 1.0);
      gl.uniform1f(uniformLocations.uSpeed, props.speed || 1.0);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", handleResize);
    };
  }, [webglState, canvasRef, props]);
};

// --- Shader Background Component ---
export const ShaderBackground = (props: { hue: number; complexity: number; speed: number }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useWebGLShader(canvasRef, props);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full object-cover z-0 pointer-events-none opacity-90"
      style={{ display: "block" }}
    />
  );
};

export interface LiquidMetalVortexProps {
  title: string;
  description: string;
  ctaButtons: { text: string; href?: string; onClick?: () => void; primary?: boolean }[];
  shaderProps: { hue: number; complexity: number; speed: number };
}

export const Hero = ({ title, description, ctaButtons, shaderProps }: LiquidMetalVortexProps) => {
  return (
    <section className="relative min-h-[90vh] w-full flex items-center justify-center overflow-hidden py-24 bg-slate-950">
      {/* Real-time WebGL Liquid Metal Canvas */}
      <ShaderBackground {...shaderProps} />

      {/* Subtle glass vignette overlays for readability */}
      <div className="absolute inset-0 bg-radial from-transparent via-slate-950/40 to-slate-950/90 pointer-events-none z-10" />
      <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-slate-950 to-transparent pointer-events-none z-10" />

      {/* Content Container */}
      <div className="relative z-20 flex flex-col items-center text-center text-white px-6 max-w-5xl mx-auto space-y-8">
        
        {/* Glow Tag */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-500/20 border border-emerald-400/40 text-emerald-300 text-xs sm:text-sm font-semibold tracking-wider backdrop-blur-md shadow-[0_0_25px_rgba(16,185,129,0.3)]">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
          <span>PROTein-conditioned Equivariant Universal SBDD</span>
        </div>

        {/* Hero Title with Chrome Glow */}
        <h1 className="text-6xl sm:text-7xl lg:text-8xl font-black tracking-tight leading-none text-transparent bg-clip-text bg-gradient-to-b from-white via-slate-100 to-slate-300 drop-shadow-[0_10px_35px_rgba(0,0,0,0.8)]">
          {title}
        </h1>

        {/* Hero Subtitle */}
        <p className="max-w-3xl text-lg sm:text-2xl text-slate-200 font-light leading-relaxed drop-shadow-md">
          {description}
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-5 pt-4">
          {ctaButtons.map((button, index) => (
            <button
              key={index}
              onClick={button.onClick}
              className={`rounded-full px-8 py-4 text-sm sm:text-base font-bold tracking-wide transition-all duration-300 cursor-pointer shadow-xl ${
                button.primary
                  ? "bg-emerald-400 text-slate-950 hover:bg-emerald-300 hover:scale-105 shadow-[0_0_35px_rgba(16,185,129,0.5)] border border-emerald-300"
                  : "bg-slate-900/80 border border-slate-700 text-slate-100 hover:bg-slate-800/90 hover:border-slate-500 hover:scale-105 backdrop-blur-md"
              }`}
            >
              {button.text}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Hero;
