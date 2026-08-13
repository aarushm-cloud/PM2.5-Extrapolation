// AERIA — DFW air quality 3D scene · 30×30 = 900 cells
// Cinematic dusk: deep teal-magenta atmosphere, warm city lights, drifting AQI haze.

// Globals from UMD: THREE

// ---------- Palette tokens ----------
const SKY_TOP    = new THREE.Color('#0a0d1a');   // deep night above
const SKY_HORIZ  = new THREE.Color('#3a2745');   // magenta haze at horizon
const SKY_GLOW   = new THREE.Color('#d97a4e');   // sun-warmed band
const FOG_NEAR   = new THREE.Color('#2a2238');
const FOG_FAR    = new THREE.Color('#4a3050');
const GROUND_BASE= new THREE.Color('#2c2740');

// ---------- AQI config ----------
const AQI = [
  { key:'good',           label:'Good',            color:'#7ed4a3', max:12.0 },
  { key:'moderate',       label:'Moderate',        color:'#ffd166', max:35.4 },
  { key:'sensitive',      label:'Unhealthy/Sens.', color:'#ff9354', max:55.4 },
  { key:'unhealthy',      label:'Unhealthy',       color:'#ef5b5b', max:150.4 },
  { key:'very_unhealthy', label:'Very Unhealthy',  color:'#a06bff', max:250.4 },
  { key:'hazardous',      label:'Hazardous',       color:'#7a2233', max:9999 },
];
function aqiFor(pm){ for (const a of AQI) if (pm <= a.max) return a; return AQI[AQI.length-1]; }
function aqiIndex(pm){ for (let i=0;i<AQI.length;i++) if (pm <= AQI[i].max) return i; return AQI.length-1; }
function aqiUS(pm){
  const bp=[[0,12,0,50],[12.1,35.4,51,100],[35.5,55.4,101,150],[55.5,150.4,151,200],[150.5,250.4,201,300],[250.5,500,301,500]];
  for (const [cl,ch,il,ih] of bp) if (pm>=cl && pm<=ch) return Math.round((ih-il)/(ch-cl)*(pm-cl)+il);
  return 500;
}

// ---------- Grid ----------
const BBOX = { n:33.08, s:32.55, e:-96.46, w:-97.05 };
const COLS = 30, ROWS = 30, TOTAL = COLS*ROWS;
const GRID_W = 360;
const GRID_H = GRID_W * (BBOX.n - BBOX.s) / (BBOX.e - BBOX.w) * Math.cos(32.78*Math.PI/180);
const CELL_W = GRID_W / COLS;
const CELL_H = GRID_H / ROWS;
// Wind direction (matches HUD: 184° = blowing toward N from S)
const WIND_RAD = (184 - 90) * Math.PI/180;
const WIND = new THREE.Vector2(Math.cos(WIND_RAD), Math.sin(WIND_RAD));

const NEIGHBORHOODS = [
  { name:'Lake Worth',         lat:32.81, lon:-97.45 },
  { name:'Fort Worth',         lat:32.75, lon:-97.33 },
  { name:'Arlington',          lat:32.73, lon:-97.10 },
  { name:'Irving',             lat:32.83, lon:-96.94 },
  { name:'DFW Airport',        lat:32.90, lon:-97.04 },
  { name:'Coppell',            lat:32.96, lon:-96.99 },
  { name:'Plano · Frisco',     lat:33.04, lon:-96.74 },
  { name:'Richardson',         lat:32.95, lon:-96.73 },
  { name:'Garland',            lat:32.91, lon:-96.62 },
  { name:'Mesquite',           lat:32.78, lon:-96.59 },
  { name:'Downtown Dallas',    lat:32.78, lon:-96.80 },
  { name:'Oak Cliff',          lat:32.71, lon:-96.84 },
  { name:'Cedar Hill',         lat:32.59, lon:-96.96 },
  { name:'Mansfield',          lat:32.57, lon:-97.13 },
  { name:'Seagoville',         lat:32.64, lon:-96.55 },
];

const SENSORS = [];
{
  let s = 12345;
  const rand = ()=>{ s=(s*1664525+1013904223)|0; return ((s>>>0)/4294967296); };
  for (let i=0;i<19;i++){
    SENSORS.push({
      lat: BBOX.s + rand()*(BBOX.n-BBOX.s),
      lon: BBOX.w + rand()*(BBOX.e-BBOX.w),
      pm:  6 + rand()*42,
    });
  }
}

function llToWorld(lat, lon){
  const fx = (lon - BBOX.w)/(BBOX.e - BBOX.w);
  const fy = (BBOX.n - lat)/(BBOX.n - BBOX.s);
  return { x: -GRID_W/2 + fx*GRID_W, z: -GRID_H/2 + fy*GRID_H };
}
function cellCenter(c, r){
  return {
    x: -GRID_W/2 + (c+0.5)*CELL_W,
    z: -GRID_H/2 + (r+0.5)*CELL_H,
    lat: BBOX.n - (r+0.5)/ROWS * (BBOX.n - BBOX.s),
    lon: BBOX.w + (c+0.5)/COLS * (BBOX.e - BBOX.w),
  };
}

function fakeZipFor(lat, lon){
  const isFW = lon < -97.0 || (lon < -96.95 && lat < 32.75);
  const binX = Math.floor((lon - BBOX.w) / ((BBOX.e - BBOX.w)/COLS) / 3);
  const binY = Math.floor((BBOX.n - lat) / ((BBOX.n - BBOX.s)/ROWS) / 3);
  const seed = binX*131 + binY*977;
  const code = ((Math.abs(seed*2654435761)|0) % 80) + 1;
  if (isFW) return `761${String(code%99).padStart(2,'0')}`;
  return `752${String(code%99).padStart(2,'0')}`;
}

function fieldPM(lat, lon){
  const LON_C = Math.cos(32.78*Math.PI/180);
  let num=0, den=0;
  for (const s of SENSORS){
    const dx = (lon - s.lon)*LON_C;
    const dy = (lat - s.lat);
    const d2 = dx*dx + dy*dy + 1e-6;
    const w = 1/Math.pow(d2, 1.5);
    num += w*s.pm; den += w;
  }
  let v = num/den;
  const dDal = Math.hypot((lon - -96.80)*LON_C, (lat - 32.78));
  v += Math.max(0, 8 - dDal*70);
  const dSD = Math.hypot((lon - -96.78)*LON_C, (lat - 32.66));
  v += Math.max(0, 12 - dSD*90);
  return Math.max(2, v);
}

function coverageFor(lat, lon){
  const LON_C = Math.cos(32.78*Math.PI/180);
  let nearest = Infinity;
  for (const s of SENSORS){
    const d = Math.hypot((lon-s.lon)*LON_C, lat-s.lat);
    if (d < nearest) nearest = d;
  }
  if (nearest < 0.04) return { label:'Good — sensor nearby', conf:'High confidence' };
  if (nearest < 0.10) return { label:'Moderate — interpolated', conf:'Medium confidence' };
  return { label:'Limited — estimated', conf:'Low confidence' };
}

const CELLS = new Array(TOTAL);
for (let r=0;r<ROWS;r++){
  for (let c=0;c<COLS;c++){
    const i = r*COLS+c;
    const cc = cellCenter(c,r);
    const pm = fieldPM(cc.lat, cc.lon);
    const cov = coverageFor(cc.lat, cc.lon);
    CELLS[i] = {
      i, c, r, x:cc.x, z:cc.z, lat:cc.lat, lon:cc.lon, pm,
      zip: fakeZipFor(cc.lat, cc.lon),
      coverage: cov.label, conf: cov.conf,
      aqiIdx: aqiIndex(pm),
    };
  }
}

// ---------- DOM ----------
const $ = (s)=>document.querySelector(s);
const canvas = $('#three-canvas');
const stage = $('#stage');
const tipEl = $('#hover-tip');
const tabs = $('#view-tabs');

// ---------- Three setup ----------
const renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.75;

const scene = new THREE.Scene();
scene.background = SKY_TOP;
scene.fog = new THREE.Fog(FOG_FAR, 700, 1800);

const cityCam = new THREE.PerspectiveCamera(42, 1, 0.5, 2400);
let camTarget = new THREE.Vector3(0, 0, 0);
cityCam.position.set(0, 180, 200);
cityCam.lookAt(camTarget);

const fpCam = new THREE.PerspectiveCamera(72, 1, 0.05, 800);
fpCam.position.set(0, 1.7, 0);
fpCam.lookAt(0, 1.7, -1);

let activeCam = cityCam;

// Cinematic dusk lights
const hemi = new THREE.HemisphereLight(0xe0b59a, 0x4a3a4e, 1.05);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffc88a, 1.85);
sun.position.set(120, 180, 200);
scene.add(sun);
const moon = new THREE.DirectionalLight(0x9ab0ff, 0.55);
moon.position.set(-140, 200, 120);
scene.add(moon);
scene.add(new THREE.AmbientLight(0xffffff, 0.18));

const cityGroup = new THREE.Group(); scene.add(cityGroup);
const fpGroup = new THREE.Group(); fpGroup.visible = false; scene.add(fpGroup);

// ---------- SKY DOME with gradient ----------
{
  const skyGeo = new THREE.SphereGeometry(1500, 32, 24);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uTop:    { value: SKY_TOP },
      uHoriz:  { value: SKY_HORIZ },
      uGlow:   { value: SKY_GLOW },
      uSunDir: { value: new THREE.Vector3(0.55, 0.18, -0.4).normalize() },
    },
    vertexShader: `varying vec3 vWorld; void main(){ vWorld = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `
      varying vec3 vWorld; uniform vec3 uTop, uHoriz, uGlow, uSunDir;
      void main(){
        float h = clamp(vWorld.y * 0.6 + 0.5, 0.0, 1.0);
        vec3 c = mix(uHoriz, uTop, smoothstep(0.0, 0.65, h));
        float sd = max(0.0, dot(vWorld, uSunDir));
        c += uGlow * pow(sd, 6.0) * 0.7;
        c += uHoriz * pow(1.0 - h, 3.0) * 0.5;
        gl_FragColor = vec4(c, 1.0);
      }`,
  });
  scene.add(new THREE.Mesh(skyGeo, skyMat));
}

// ---------- GROUND APRON with radial vignette ----------
{
  const apronMat = new THREE.ShaderMaterial({
    uniforms: { uBase: { value: GROUND_BASE }, uEdge: { value: FOG_FAR } },
    vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: `
      varying vec2 vUv; uniform vec3 uBase, uEdge;
      void main(){
        float d = distance(vUv, vec2(0.5));
        vec3 c = mix(uBase * 1.4, uEdge * 0.9, smoothstep(0.12, 0.6, d));
        gl_FragColor = vec4(c, 1.0);
      }`,
  });
  const apron = new THREE.Mesh(new THREE.PlaneGeometry(GRID_W*4, GRID_H*4), apronMat);
  apron.rotation.x = -Math.PI/2; apron.position.y = -0.3;
  cityGroup.add(apron);
}

// ---------- CELL FLOOR (instanced) — soft AQI tint ----------
const cellGeo = new THREE.PlaneGeometry(CELL_W*0.97, CELL_H*0.97); cellGeo.rotateX(-Math.PI/2);
const cellMat = new THREE.MeshStandardMaterial({ roughness:1.0, metalness:0.0, vertexColors:false });
const cellMesh = new THREE.InstancedMesh(cellGeo, cellMat, TOTAL);
cellMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
const cellColors = new Float32Array(TOTAL*3);
const baseFloor = new THREE.Color('#3a3550');
const dummy = new THREE.Object3D();
for (const cell of CELLS){
  dummy.position.set(cell.x, 0.02, cell.z);
  dummy.updateMatrix();
  cellMesh.setMatrixAt(cell.i, dummy.matrix);
  const a = AQI[cell.aqiIdx];
  const tint = new THREE.Color(a.color);
  const col = baseFloor.clone().lerp(tint, 0.10);
  cellColors[cell.i*3+0] = col.r;
  cellColors[cell.i*3+1] = col.g;
  cellColors[cell.i*3+2] = col.b;
}
cellMesh.instanceColor = new THREE.InstancedBufferAttribute(cellColors, 3);
cellMat.vertexColors = true;
cellMesh.instanceMatrix.needsUpdate = true;
cityGroup.add(cellMesh);

// ---------- GRIDLINES — thin, only visible nearby (depth fade) ----------
{
  const verts = []; const Y = 0.06;
  for (let i=0;i<=COLS;i++){
    const x = -GRID_W/2 + i*CELL_W;
    verts.push(x, Y, -GRID_H/2,  x, Y, GRID_H/2);
  }
  for (let j=0;j<=ROWS;j++){
    const z = -GRID_H/2 + j*CELL_H;
    verts.push(-GRID_W/2, Y, z,  GRID_W/2, Y, z);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  const m = new THREE.LineBasicMaterial({ color:0x2a3050, transparent:true, opacity:0.18 });
  cityGroup.add(new THREE.LineSegments(g, m));
}

// ---------- HOVER + SELECT FRAMES ----------
const hoverGeo = new THREE.EdgesGeometry(new THREE.PlaneGeometry(CELL_W*0.97, CELL_H*0.97));
const hoverMat = new THREE.LineBasicMaterial({ color:0xfff1c8, transparent:true, opacity:0.85 });
const hoverFrame = new THREE.LineSegments(hoverGeo, hoverMat);
hoverFrame.rotation.x = -Math.PI/2; hoverFrame.position.y = 0.10; hoverFrame.visible = false;
cityGroup.add(hoverFrame);

// Selection: glowing disc + lifted ring
const selectRingMat = new THREE.MeshBasicMaterial({ color:0xfff1c8, transparent:true, opacity:0.95, depthWrite:false });
const selectRing = new THREE.Mesh(
  new THREE.RingGeometry(Math.min(CELL_W,CELL_H)*0.4, Math.min(CELL_W,CELL_H)*0.5, 48),
  selectRingMat
);
selectRing.rotation.x = -Math.PI/2; selectRing.position.y = 0.12;
cityGroup.add(selectRing);

const selectGlowMat = new THREE.MeshBasicMaterial({ color:0xffd166, transparent:true, opacity:0.32, depthWrite:false, blending:THREE.AdditiveBlending });
const selectGlow = new THREE.Mesh(
  new THREE.CircleGeometry(Math.min(CELL_W,CELL_H)*0.55, 32),
  selectGlowMat
);
selectGlow.rotation.x = -Math.PI/2; selectGlow.position.y = 0.08;
cityGroup.add(selectGlow);

// ---------- BUILDINGS — density centers per real DFW geography ----------
// intensity: 0..1 weight, maxH: max storey-equiv height in world units, decay: km falloff
const DENSITY_CENTERS = [
  { name:'Dallas downtown', lat:32.776, lon:-96.797, intensity:1.00, maxH:46, decay:0.040 },
  { name:'Plano',           lat:33.020, lon:-96.699, intensity:0.62, maxH:18, decay:0.050 },
  { name:'Garland',         lat:32.912, lon:-96.639, intensity:0.45, maxH:11, decay:0.055 },
  { name:'Irving',          lat:32.814, lon:-96.940, intensity:0.55, maxH:16, decay:0.055 },
  { name:'Richardson',      lat:32.948, lon:-96.730, intensity:0.45, maxH:12, decay:0.055 },
  { name:'Carrollton',      lat:32.954, lon:-96.895, intensity:0.32, maxH:8,  decay:0.060 },
  { name:'Mesquite',        lat:32.767, lon:-96.599, intensity:0.30, maxH:7,  decay:0.060 },
  { name:'Lewisville',      lat:33.046, lon:-96.994, intensity:0.22, maxH:6,  decay:0.065 },
  { name:'Cedar Hill',      lat:32.59,  lon:-96.96,  intensity:0.18, maxH:5,  decay:0.070 },
  { name:'Arlington',       lat:32.73,  lon:-97.10,  intensity:0.40, maxH:9,  decay:0.060 },
  { name:'Fort Worth',      lat:32.755, lon:-97.330, intensity:0.65, maxH:22, decay:0.045 },
];
const LON_C = Math.cos(32.78*Math.PI/180);
function densityAt(lat, lon){
  let dens = 0, maxH = 0, near = Infinity;
  for (const c of DENSITY_CENTERS){
    const dx = (lon - c.lon)*LON_C;
    const dy = lat - c.lat;
    const d = Math.hypot(dx, dy);
    const w = Math.exp(-(d*d)/(c.decay*c.decay)) * c.intensity;
    dens += w;
    maxH = Math.max(maxH, c.maxH * Math.exp(-(d*d)/(c.decay*c.decay*1.4)));
    if (d < near) near = d;
  }
  return { dens: Math.min(1.4, dens), maxH, near };
}

// Pre-compute per-cell density/max-height
const CELL_BLDG = CELLS.map(cell=>{
  const d = densityAt(cell.lat, cell.lon);
  return { ...d };
});

const perCellCount = [];
let totalBldg = 0;
function rngOf(seed){ let s = seed|0; return ()=>{ s=(s*1664525+1013904223)|0; return ((s>>>0)/4294967296); }; }
for (const cell of CELLS){
  const cb = CELL_BLDG[cell.i];
  // Building count: dense urban = many, suburban = few, edges = sparse
  let n;
  if (cb.dens > 0.85) n = 28 + Math.floor(cb.dens*8);     // dense urban core
  else if (cb.dens > 0.45) n = 16 + Math.floor(cb.dens*10); // mid-rise cluster
  else if (cb.dens > 0.20) n = 8 + Math.floor(cb.dens*8); // suburban
  else if (cb.dens > 0.08) n = 4 + Math.floor(cb.dens*12); // sparse
  else n = (cell.c + cell.r) % 2 === 0 ? 2 : 1;           // near-empty edges
  perCellCount.push(n);
  totalBldg += n;
}

const bldgGeo = new THREE.BoxGeometry(1, 1, 1);
const bldgMat = new THREE.MeshStandardMaterial({ vertexColors:true, roughness:0.85, metalness:0.05 });
const bldgMesh = new THREE.InstancedMesh(bldgGeo, bldgMat, totalBldg);
bldgMesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
const bldgColors = new Float32Array(totalBldg*3);

// Window glow points — separate Points cloud, additive
const winPositions = [];
{
  let idx = 0;
  for (const cell of CELLS){
    const n = perCellCount[cell.i];
    if (n === 0) continue;
    const cb = CELL_BLDG[cell.i];
    const rand = rngOf(cell.c*131 + cell.r*977 + 7);

    for (let i=0;i<n;i++){
      // Footprint scales with density: downtown = wide towers, suburbs = small homes
      let w, d;
      if (cb.dens > 0.85){      // urban core
        w = 3.0 + rand()*2.5;
        d = 3.0 + rand()*2.5;
      } else if (cb.dens > 0.45){ // mid-rise
        w = 2.5 + rand()*2.2;
        d = 2.5 + rand()*2.2;
      } else if (cb.dens > 0.15){ // suburban
        w = 1.8 + rand()*1.6;
        d = 1.8 + rand()*1.6;
      } else {                   // residential / sparse
        w = 1.2 + rand()*1.0;
        d = 1.2 + rand()*1.0;
      }

      // Height: power-law within max so skyscrapers feel rare and dramatic
      const tProb = rand();
      let h;
      if (cb.dens > 0.85){
        // urban core: long-tail toward maxH
        const tower = Math.pow(tProb, 2.5);
        h = 3 + tower * cb.maxH;
      } else if (cb.dens > 0.45){
        h = 2 + Math.pow(tProb, 2.0) * cb.maxH * 0.85;
      } else if (cb.dens > 0.15){
        h = 1.0 + Math.pow(tProb, 1.6) * cb.maxH * 0.6;
      } else {
        h = 0.6 + rand()*1.8;
      }

      const px = cell.x + (rand()-0.5)*CELL_W*0.7;
      const pz = cell.z + (rand()-0.5)*CELL_H*0.7;
      dummy.position.set(px, h/2, pz);
      dummy.scale.set(w, h, d);
      dummy.rotation.set(0,0,0);
      dummy.updateMatrix();
      bldgMesh.setMatrixAt(idx, dummy.matrix);

      // Color: warm-tinted concrete that reads against the dark ground.
      // Brighter base + strong sun catch on tall faces.
      const slate = 0.78 + rand()*0.14;
      const r0 = slate * 1.05;
      const g0 = slate * 0.98;
      const b0 = slate * 1.02;
      // All buildings catch some warm dusk light; tall ones catch strongly
      const sunCatch = 0.5 + Math.min(0.5, h / 18);
      bldgColors[idx*3+0] = Math.min(1.0, r0 + sunCatch * 0.32);
      bldgColors[idx*3+1] = Math.min(1.0, g0 + sunCatch * 0.18);
      bldgColors[idx*3+2] = Math.min(1.0, b0 + sunCatch * 0.04);

      // Window glow — only on mid/tall buildings, denser on tall ones
      if (h > 3.5){
        const winCount = Math.min(80, Math.floor(h * (cb.dens > 0.85 ? 3.5 : 2.0)));
        for (let wi=0; wi<winCount; wi++){
          if (rand() > (cb.dens > 0.85 ? 0.42 : 0.6)) continue;
          const wy = 1.2 + rand()*(h - 1.8);
          const face = Math.floor(rand()*4);
          let wx = px, wz = pz;
          if (face === 0){ wx += w/2 + 0.01; wz += (rand()-0.5)*d*0.85; }
          else if (face === 1){ wx -= w/2 + 0.01; wz += (rand()-0.5)*d*0.85; }
          else if (face === 2){ wz += d/2 + 0.01; wx += (rand()-0.5)*w*0.85; }
          else { wz -= d/2 + 0.01; wx += (rand()-0.5)*w*0.85; }
          winPositions.push(wx, wy, wz);
        }
      }
      idx++;
    }
  }
}
bldgMesh.instanceColor = new THREE.InstancedBufferAttribute(bldgColors, 3);
bldgMat.vertexColors = true;
bldgMesh.instanceMatrix.needsUpdate = true;
cityGroup.add(bldgMesh);

// Window glow
{
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(winPositions, 3));
  // each window gets a slight color jitter
  const colors = new Float32Array(winPositions.length);
  for (let i=0;i<colors.length;i+=3){
    const t = Math.random();
    if (t < 0.7){ colors[i]=1.0; colors[i+1]=0.78; colors[i+2]=0.46; }       // warm
    else if (t < 0.92){ colors[i]=0.98; colors[i+1]=0.86; colors[i+2]=0.62; } // pale warm
    else { colors[i]=0.62; colors[i+1]=0.78; colors[i+2]=1.0; }              // cool blue
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.PointsMaterial({
    size: 0.35, vertexColors:true, transparent:true, opacity:0.9,
    blending: THREE.AdditiveBlending, depthWrite:false, sizeAttenuation:true
  });
  const pts = new THREE.Points(geo, mat);
  cityGroup.add(pts);
}

// ---------- HIGHWAYS — glowing lines suggesting traffic ----------
{
  // a simple curved polyline approximating I-30 / I-35E
  function makeStream(points, color, width){
    const g = new THREE.BufferGeometry();
    const verts = [];
    for (let i=0;i<points.length-1;i++){
      verts.push(...points[i], ...points[i+1]);
    }
    g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    const m = new THREE.LineBasicMaterial({ color, transparent:true, opacity:0.6, blending:THREE.AdditiveBlending, depthWrite:false });
    const line = new THREE.LineSegments(g, m);
    cityGroup.add(line);

    // Thicker glow plate beneath
    for (let i=0;i<points.length-1;i++){
      const [x1,y1,z1] = points[i], [x2,,z2] = points[i+1];
      const mx=(x1+x2)/2, mz=(z1+z2)/2;
      const dx=x2-x1, dz=z2-z1;
      const len = Math.hypot(dx, dz);
      const ang = Math.atan2(dz, dx);
      const plate = new THREE.Mesh(
        new THREE.PlaneGeometry(len, width),
        new THREE.MeshBasicMaterial({ color, transparent:true, opacity:0.18, blending:THREE.AdditiveBlending, depthWrite:false })
      );
      plate.rotation.x = -Math.PI/2; plate.rotation.z = -ang;
      plate.position.set(mx, 0.07, mz);
      cityGroup.add(plate);
    }
  }
  // I-30 (E-W through downtown)
  const i30 = [];
  for (let i=0;i<=20;i++){
    const t = i/20;
    const x = -GRID_W/2 + t*GRID_W;
    const z = -8 + Math.sin(t*Math.PI)*8;
    i30.push([x, 0.08, z]);
  }
  makeStream(i30, 0xff9354, 2.2);

  // I-35E (NE-SW through Dallas)
  const i35 = [];
  for (let i=0;i<=20;i++){
    const t = i/20;
    const x = -10 + (t-0.5)*GRID_W*0.6;
    const z = -GRID_H/2 + t*GRID_H;
    i35.push([x, 0.08, z]);
  }
  makeStream(i35, 0xffd166, 1.8);
}

// ---------- DASHED BBOX OUTLINE ----------
{
  const ext = 1.04;
  const w = GRID_W*ext, h = GRID_H*ext;
  const pts = [
    new THREE.Vector3(-w/2, 0.2, -h/2),
    new THREE.Vector3( w/2, 0.2, -h/2),
    new THREE.Vector3( w/2, 0.2,  h/2),
    new THREE.Vector3(-w/2, 0.2,  h/2),
    new THREE.Vector3(-w/2, 0.2, -h/2),
  ];
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineDashedMaterial({ color:0xffd166, dashSize:3.5, gapSize:2.5, transparent:true, opacity:0.55 });
  const line = new THREE.Line(g, mat);
  line.computeLineDistances();
  cityGroup.add(line);
  const tickMat = new THREE.LineBasicMaterial({ color:0xffd166, transparent:true, opacity:0.85 });
  const tickLen = 6;
  [[-w/2,-h/2,1,1],[w/2,-h/2,-1,1],[w/2,h/2,-1,-1],[-w/2,h/2,1,-1]].forEach(([x,z,sx,sz])=>{
    const p1 = [new THREE.Vector3(x,0.22,z), new THREE.Vector3(x+sx*tickLen,0.22,z)];
    const p2 = [new THREE.Vector3(x,0.22,z), new THREE.Vector3(x,0.22,z+sz*tickLen)];
    cityGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p1), tickMat));
    cityGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p2), tickMat));
  });
}

// ---------- AQI PARTICLE CLOUDS — drift with wind, glow, parallax ----------
const particleGroups = [];
function buildParticleClouds(){
  const buckets = AQI.map(()=>[]);
  for (const cell of CELLS) buckets[cell.aqiIdx].push(cell);

  for (let k=0;k<AQI.length;k++){
    const cells = buckets[k];
    if (cells.length === 0) { particleGroups.push(null); continue; }
    const severity = (k+1)/AQI.length;
    const perCell = Math.round(10 + severity*40);
    const total = Math.min(cells.length * perCell, 5000);
    const positions = new Float32Array(total*3);
    const phases = new Float32Array(total);
    const seeds = new Float32Array(total);
    const sizes = new Float32Array(total);
    for (let i=0;i<total;i++){
      const cell = cells[Math.floor(Math.random()*cells.length)];
      const ang = Math.random()*Math.PI*2;
      const r = Math.pow(Math.random(),0.55) * Math.min(CELL_W, CELL_H)*0.5;
      const baseY = 4 + severity*5 + Math.random()*7;
      positions[i*3+0] = cell.x + Math.cos(ang)*r;
      positions[i*3+1] = baseY;
      positions[i*3+2] = cell.z + Math.sin(ang)*r;
      phases[i] = Math.random()*Math.PI*2;
      seeds[i] = 0.4 + Math.random()*0.9;
      sizes[i] = 0.6 + Math.random()*1.4;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));
    geo.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
    geo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
    const mat = new THREE.PointsMaterial({
      color: new THREE.Color(AQI[k].color),
      size: 0.55 + severity*0.85,
      transparent:true,
      opacity: 0.42 + severity*0.42,
      depthWrite:false,
      blending:THREE.AdditiveBlending,
      sizeAttenuation:true,
    });
    const pts = new THREE.Points(geo, mat);
    pts.userData.basePositions = positions.slice();
    cityGroup.add(pts);
    particleGroups.push(pts);
  }
}
buildParticleClouds();

// ---------- HORIZON HAZE — large additive plane bands ----------
{
  for (let i=0; i<3; i++){
    const w = GRID_W*2.2;
    const h = 30 + i*20;
    const dist = 280 + i*110;
    const mat = new THREE.MeshBasicMaterial({
      color: i===0 ? 0xa06bff : (i===1 ? 0xff9354 : 0x3a2745),
      transparent:true, opacity: 0.10 - i*0.025,
      blending:THREE.AdditiveBlending, depthWrite:false, side:THREE.DoubleSide
    });
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
    m.position.set(0, h/2 + 4, -dist);
    cityGroup.add(m);
    const m2 = m.clone(); m2.position.set(0, h/2+4, dist); cityGroup.add(m2);
    const m3 = m.clone(); m3.rotation.y = Math.PI/2; m3.position.set(-dist, h/2+4, 0); cityGroup.add(m3);
    const m4 = m3.clone(); m4.position.set(dist, h/2+4, 0); cityGroup.add(m4);
  }
}

// ---------- LABEL LAYER ----------
const labelLayer = document.createElement('div');
labelLayer.style.cssText = 'position:absolute;inset:0;pointer-events:none;z-index:7;';
stage.appendChild(labelLayer);
const labelEls = [];

const ZIP_LABELS = [];
for (let r=1;r<ROWS;r+=3){
  for (let c=1;c<COLS;c+=3){
    const cell = CELLS[r*COLS + c];
    const el = document.createElement('div');
    el.className = 'zip-label';
    el.style.cssText = `
      position:absolute;transform:translate(-50%,-50%);
      font-family:"JetBrains Mono",monospace;font-size:9.5px;
      color:rgba(255,232,205,0.55);
      letter-spacing:0.04em;
      pointer-events:none;
      white-space:nowrap;
      text-shadow:0 1px 3px rgba(0,0,0,0.85);
    `;
    el.textContent = cell.zip;
    labelLayer.appendChild(el);
    ZIP_LABELS.push({ el, x:cell.x, y:1.6, z:cell.z });
  }
}

const NEIGH_LABELS = [];
for (const n of NEIGHBORHOODS){
  const w = llToWorld(n.lat, n.lon);
  const el = document.createElement('div');
  el.style.cssText = `
    position:absolute;transform:translate(-50%,-50%);
    font-family:"Inter Tight","Inter",sans-serif;
    font-size:11.5px;font-weight:500;letter-spacing:0.005em;
    color:#f3eadc;
    background:rgba(20,16,32,0.78);
    border:1px solid rgba(255,232,205,0.16);
    padding:3px 9px;border-radius:4px;
    backdrop-filter:blur(8px);
    pointer-events:none;
    white-space:nowrap;
    box-shadow:0 4px 14px rgba(0,0,0,0.5);
  `;
  el.textContent = n.name;
  labelLayer.appendChild(el);
  NEIGH_LABELS.push({ el, x:w.x, y:7, z:w.z });
}

{
  const ext = 1.04;
  const h = GRID_H*ext;
  const el = document.createElement('div');
  el.style.cssText = `
    position:absolute;transform:translate(-50%,-50%);
    font-family:"JetBrains Mono",monospace;font-size:9.5px;
    color:#ffd166;letter-spacing:0.12em;text-transform:uppercase;
    background:rgba(20,16,32,0.78);
    border:1px solid rgba(255,209,102,0.4);
    padding:3px 8px;border-radius:4px;
    pointer-events:none;
  `;
  el.textContent = 'data bbox · 32.55 → 33.08 · −97.05 → −96.46';
  labelLayer.appendChild(el);
  labelEls.push({ el, x:0, y:0.4, z:-h/2 - 5 });
}

// ---------- FIRST-PERSON SCENE ----------
let fpParticleSystem = null;
{
  // Wet street with reflective material
  const g = new THREE.Mesh(new THREE.PlaneGeometry(420,420),
    new THREE.MeshStandardMaterial({ color:0x141520, roughness:0.85, metalness:0.05 }));
  g.rotation.x = -Math.PI/2; fpGroup.add(g);
  const street = new THREE.Mesh(new THREE.PlaneGeometry(8, 420),
    new THREE.MeshStandardMaterial({ color:0x0c0d18, roughness:0.55, metalness:0.15 }));
  street.rotation.x = -Math.PI/2; street.position.y = 0.001; fpGroup.add(street);
  for (let z=-200; z<200; z+=8){
    const dash = new THREE.Mesh(new THREE.PlaneGeometry(0.2,2.2),
      new THREE.MeshBasicMaterial({ color:0x6a5a3a }));
    dash.rotation.x = -Math.PI/2; dash.position.set(0, 0.005, z); fpGroup.add(dash);
  }
  [-5.4, 5.4].forEach(x=>{
    const sw = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 420),
      new THREE.MeshStandardMaterial({ color:0x1c1d2a, roughness:0.95 }));
    sw.rotation.x = -Math.PI/2; sw.position.set(x, 0.002, 0); fpGroup.add(sw);
  });
  const rand = rngOf(2026);
  const winGeo = new THREE.PlaneGeometry(0.6, 0.9);
  const warmWin = new THREE.MeshBasicMaterial({ color:0xffb778, transparent:true, opacity:0.92 });
  const dimWin = new THREE.MeshBasicMaterial({ color:0xffe9c2, transparent:true, opacity:0.7 });
  const coolWin = new THREE.MeshBasicMaterial({ color:0x9bc4ff, transparent:true, opacity:0.85 });
  for (const side of [-1, 1]){
    const xBase = side*12;
    let z = -200;
    while (z < 200){
      const w = 6 + rand()*10;
      const d = 8 + rand()*16;
      const h = 10 + rand()*55;
      const shade = 0.10 + rand()*0.16;
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(w, h, d),
        new THREE.MeshStandardMaterial({ color:new THREE.Color(shade,shade,shade*1.12), roughness:0.75, metalness:0.12 })
      );
      m.position.set(xBase + side*(w/2 + rand()*4), h/2, z + d/2);
      fpGroup.add(m);
      // Window glow grid
      const cols = Math.max(2, Math.floor(w/1.2));
      const rows = Math.max(3, Math.floor(h/2.0));
      for (let cc=0;cc<cols;cc++){
        for (let rr=0;rr<rows;rr++){
          if (rand() > 0.45) continue;
          const matChoice = rand();
          const mat = matChoice < 0.65 ? warmWin : (matChoice < 0.92 ? dimWin : coolWin);
          const ww = new THREE.Mesh(winGeo, mat);
          const lx = (cc+0.5)/cols * w*0.85 - w*0.85/2;
          const ly = 1.5 + (rr+0.5)/rows * (h - 3);
          const faceZ = z + d/2 + (side>0 ? -d/2 - 0.02 : d/2 + 0.02);
          ww.position.set(xBase + side*(w/2 + rand()*0.05) + (side<0 ? lx : lx), ly, faceZ);
          if (side<0) ww.rotation.y = Math.PI;
          fpGroup.add(ww);
        }
      }
      z += d + 1.2 + rand()*2;
    }
  }
  // Crosswalks
  for (const cz of [-60, 60]){
    const cross = new THREE.Mesh(new THREE.PlaneGeometry(420,6),
      new THREE.MeshStandardMaterial({ color:0x0a0b14, roughness:0.5, metalness:0.18 }));
    cross.rotation.x = -Math.PI/2; cross.position.set(0, 0.001, cz); fpGroup.add(cross);
  }
  // Streetlamp posts with point lights
  for (let z=-180; z<180; z+=22){
    [-6.4, 6.4].forEach(x=>{
      const post = new THREE.Mesh(
        new THREE.CylinderGeometry(0.06, 0.06, 7, 8),
        new THREE.MeshStandardMaterial({ color:0x2a2d3a, roughness:0.7 })
      );
      post.position.set(x,3.5,z); fpGroup.add(post);
      const lampHead = new THREE.Mesh(
        new THREE.SphereGeometry(0.18, 12, 8),
        new THREE.MeshBasicMaterial({ color:0xffc97a })
      );
      lampHead.position.set(x, 7, z); fpGroup.add(lampHead);
      const halo = new THREE.Mesh(
        new THREE.SphereGeometry(1.8, 12, 8),
        new THREE.MeshBasicMaterial({ color:0xffc97a, transparent:true, opacity:0.10, blending:THREE.AdditiveBlending, depthWrite:false })
      );
      halo.position.set(x, 7, z); fpGroup.add(halo);
    });
  }
  // FP ambient + fill
  const fpAmb = new THREE.AmbientLight(0x6a4a3a, 0.4); fpGroup.add(fpAmb);
  const fpHemi = new THREE.HemisphereLight(0xffa970, 0x1a1430, 0.5); fpGroup.add(fpHemi);
}

function buildFPParticles(pm){
  if (fpParticleSystem){
    fpGroup.remove(fpParticleSystem);
    fpParticleSystem.geometry.dispose();
    fpParticleSystem.material.dispose();
  }
  const a = aqiFor(pm);
  const severity = Math.max(0.05, Math.min(1, pm/60));
  const count = Math.round(900 + severity*5000);
  const positions = new Float32Array(count*3);
  const phases = new Float32Array(count);
  const radius = 42;
  for (let i=0;i<count;i++){
    const r = Math.pow(Math.random(),0.65)*radius;
    const theta = Math.random()*Math.PI*2;
    positions[i*3+0] = Math.cos(theta)*r;
    positions[i*3+1] = 0.3 + Math.random()*16;
    positions[i*3+2] = Math.sin(theta)*r;
    phases[i] = Math.random()*Math.PI*2;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));
  const mat = new THREE.PointsMaterial({
    color:new THREE.Color(a.color),
    size:0.07 + severity*0.06,
    transparent:true,
    opacity:0.55 + severity*0.4,
    depthWrite:false,
    blending:THREE.AdditiveBlending,
    sizeAttenuation:true,
  });
  fpParticleSystem = new THREE.Points(geo, mat);
  fpGroup.add(fpParticleSystem);
  return { count, severity, a };
}

// ---------- RESIZE / LABEL PROJECTION ----------
function resize(){
  const r = stage.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  cityCam.aspect = r.width/r.height; cityCam.updateProjectionMatrix();
  fpCam.aspect = r.width/r.height; fpCam.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();

const tmpV = new THREE.Vector3();
const camDistRef = cityCam.position.distanceTo(camTarget);
function projectLabels(){
  const r = stage.getBoundingClientRect();
  const inCity = activeCam === cityCam;
  const dist = cityCam.position.distanceTo(camTarget);
  const zoom = camDistRef / Math.max(1, dist);
  const zipOpacity = Math.max(0, Math.min(0.85, (zoom - 0.85) * 1.4));

  for (const L of labelEls){
    tmpV.set(L.x, L.y, L.z); tmpV.project(activeCam);
    const x = (tmpV.x*0.5+0.5)*r.width;
    const y = (-tmpV.y*0.5+0.5)*r.height;
    L.el.style.opacity = inCity && tmpV.z < 1 ? '1' : '0';
    L.el.style.left = `${x}px`; L.el.style.top  = `${y}px`;
  }
  for (const L of NEIGH_LABELS){
    tmpV.set(L.x, L.y, L.z); tmpV.project(activeCam);
    const x = (tmpV.x*0.5+0.5)*r.width;
    const y = (-tmpV.y*0.5+0.5)*r.height;
    L.el.style.opacity = inCity && tmpV.z < 1 ? '1' : '0';
    L.el.style.left = `${x}px`; L.el.style.top  = `${y}px`;
  }
  for (const L of ZIP_LABELS){
    tmpV.set(L.x, L.y, L.z); tmpV.project(activeCam);
    const x = (tmpV.x*0.5+0.5)*r.width;
    const y = (-tmpV.y*0.5+0.5)*r.height;
    L.el.style.opacity = inCity && tmpV.z < 1 ? String(zipOpacity) : '0';
    L.el.style.left = `${x}px`; L.el.style.top  = `${y}px`;
  }
}

// ---------- INTERACTION ----------
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let hoveredCell = null, selectedCell = null;
let isDragging = false, dragStart = {x:0,y:0}, dragStartCam = new THREE.Vector3();

function pickCell(evt){
  const r = canvas.getBoundingClientRect();
  mouse.x = ((evt.clientX - r.left)/r.width)*2 - 1;
  mouse.y = -((evt.clientY - r.top)/r.height)*2 + 1;
  raycaster.setFromCamera(mouse, cityCam);
  const hits = raycaster.intersectObject(cellMesh, false);
  if (hits.length) return CELLS[hits[0].instanceId];
  return null;
}

function setHoverCell(cell, evt){
  hoveredCell = cell;
  if (cell){
    hoverFrame.position.set(cell.x, 0.10, cell.z);
    hoverFrame.visible = true;
    canvas.style.cursor = 'pointer';
    if (evt){
      const a = aqiFor(cell.pm);
      tipEl.classList.add('show');
      tipEl.innerHTML = `
        <div style="display:flex;gap:10px;align-items:center;margin-bottom:2px">
          <span style="color:#f3eadc;font-weight:500">${cell.zip}</span>
          <span style="width:6px;height:6px;border-radius:50%;background:${a.color};display:inline-block;box-shadow:0 0 6px ${a.color}"></span>
          <span style="color:#dcd0bc">${a.label}</span>
        </div>
        <div style="color:#beb3a0;font-size:10px;margin-bottom:1px">${cell.pm.toFixed(1)} µg/m³ · cell ${cell.c+1}/${cell.r+1}</div>
        <div style="color:#8a7f6e;font-size:9.5px">${cell.coverage}</div>`;
      const r = stage.getBoundingClientRect();
      tipEl.style.left = `${evt.clientX - r.left}px`;
      tipEl.style.top = `${evt.clientY - r.top}px`;
    }
  } else {
    hoverFrame.visible = false;
    tipEl.classList.remove('show');
    canvas.style.cursor = isDragging ? 'grabbing' : 'grab';
  }
}

canvas.addEventListener('mousemove', (evt)=>{
  if (activeCam !== cityCam) return;
  if (isDragging){
    const dx = (evt.clientX - dragStart.x);
    const dy = (evt.clientY - dragStart.y);
    cityCam.position.x = dragStartCam.x - dx*0.45;
    cityCam.position.z = dragStartCam.z - dy*0.45;
    camTarget.set(cityCam.position.x, 0, cityCam.position.z - 100);
    cityCam.lookAt(camTarget);
    return;
  }
  const cell = pickCell(evt);
  setHoverCell(cell, evt);
});

canvas.addEventListener('mousedown', (evt)=>{
  if (activeCam !== cityCam) return;
  dragStart = { x:evt.clientX, y:evt.clientY };
  dragStartCam.copy(cityCam.position);
  const onMove = (e)=>{
    if (Math.abs(e.clientX - dragStart.x) + Math.abs(e.clientY - dragStart.y) > 4){
      isDragging = true;
      canvas.classList.add('is-dragging');
      window.removeEventListener('mousemove', onMove);
    }
  };
  window.addEventListener('mousemove', onMove);
  const onUp = ()=>{
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', onUp);
    canvas.classList.remove('is-dragging');
    setTimeout(()=>{ isDragging = false; }, 0);
  };
  window.addEventListener('mouseup', onUp);
});

canvas.addEventListener('click', (evt)=>{
  if (activeCam !== cityCam || isDragging) return;
  const cell = pickCell(evt);
  if (cell) selectCell(cell);
});

canvas.addEventListener('wheel', (evt)=>{
  if (activeCam !== cityCam) return;
  evt.preventDefault();
  const dir = Math.sign(evt.deltaY);
  const newY = THREE.MathUtils.clamp(cityCam.position.y * (1 + dir*0.08), 60, 420);
  cityCam.position.y = newY;
  camTarget.set(cityCam.position.x, 0, cityCam.position.z - 100);
  cityCam.lookAt(camTarget);
}, { passive:false });

// ---------- SELECTION / PANEL SYNC ----------
function selectCell(cell, opts={}){
  selectedCell = cell;
  selectRing.position.set(cell.x, 0.12, cell.z);
  selectGlow.position.set(cell.x, 0.08, cell.z);
  // brighten select to match AQI
  const a = aqiFor(cell.pm);
  selectGlowMat.color.set(a.color);
  syncPanels(cell);
  if (opts.pan){
    camTarget.set(cell.x, 0, cell.z + 30);
    cityCam.position.set(cell.x, Math.min(180, Math.max(140, cityCam.position.y)), cell.z + 110);
    cityCam.lookAt(camTarget);
  }
}

function syncPanels(cell){
  const a = aqiFor(cell.pm);
  $('#lp-cat').textContent = a.label;
  const dot = $('#lp-dot'); dot.style.color = a.color; dot.style.background = a.color;
  $('#lp-pm').textContent = cell.pm.toFixed(1);
  const delta = (cell.pm - 19.0).toFixed(1);
  $('#lp-delta').textContent = `${cell.pm > 19 ? '▲' : '▼'} ${Math.abs(delta)} vs. 24h`;

  const recsByCat = {
    good: { who:'<b>Everyone</b> can be active outdoors. Sensitive groups (asthma, COPD, heart disease, elderly, children, pregnancy) face minimal risk.', ok:'Air quality is satisfactory and poses little or no health risk.', ex:'<b>Outdoor exercise</b> — no restrictions. Great evening to be outside.', win:'<b>Windows</b> — open and ventilate freely.', mask:'<b>Masks</b> — not needed.', conf:'High confidence' },
    moderate: { who:'<b>Sensitive groups</b> — asthma, COPD, heart disease, elderly, children, pregnancy may notice symptoms with prolonged or heavy exertion.', ok:'The general population is unlikely to be affected.', ex:'<b>Outdoor exercise</b> — fine for most. Sensitive groups should consider shorter or lower-intensity sessions.', win:'<b>Windows</b> — open is fine. Close briefly if you smell smoke or see haze.', mask:'<b>Masks</b> — not needed for the general public.', conf:'High confidence' },
    sensitive: { who:'<b>Sensitive groups</b> should reduce prolonged or heavy outdoor exertion. Includes asthma, COPD, heart disease, elderly, children, and pregnant people.', ok:'General public unlikely to be affected, but watch for symptoms.', ex:'<b>Outdoor exercise</b> — sensitive groups: shorten and lighten. General public: take more breaks.', win:'<b>Windows</b> — keep mostly closed during peak hours (afternoon).', mask:'<b>Masks</b> — N95 helpful for sensitive groups during outdoor activity.', conf:'Medium confidence' },
    unhealthy: { who:'<b>Sensitive groups</b> should avoid prolonged outdoor exertion. <b>Everyone</b> should reduce heavy exertion.', ok:'Health effects possible for the general public; sensitive groups face greater risk.', ex:'<b>Outdoor exercise</b> — move indoors if possible. Sensitive groups should stay in.', win:'<b>Windows</b> — closed. Run HVAC on recirculate or use a HEPA purifier.', mask:'<b>Masks</b> — N95 recommended outdoors, especially for sensitive groups.', conf:'Medium confidence' },
    very_unhealthy: { who:'<b>Sensitive groups</b> should remain indoors. <b>Everyone</b> should avoid prolonged outdoor exertion.', ok:'Health alert: significant risk of effects for the general population.', ex:'<b>Outdoor exercise</b> — cancel. Reschedule for cleaner air.', win:'<b>Windows</b> — sealed. Run HEPA filtration in occupied rooms.', mask:'<b>Masks</b> — N95 outdoors for everyone, fitted properly.', conf:'Low confidence — limited sensors' },
    hazardous: { who:'<b>Everyone</b> should remain indoors and minimize physical activity. Especially sensitive groups: asthma, COPD, heart disease, elderly, children, pregnancy.', ok:'Emergency conditions. Entire population is more likely to be affected.', ex:'<b>Outdoor exercise</b> — do not. Stay indoors.', win:'<b>Windows</b> — sealed. HEPA + recirculation. Avoid combustion sources indoors.', mask:'<b>Masks</b> — N95/P100 mandatory for any time outside.', conf:'Low confidence — limited sensors' },
  };
  const recs = recsByCat[a.key];
  $('#left-panel').querySelector('.lp-section:nth-of-type(2)').innerHTML = `
    <h4>Who should take care</h4>
    <div class="lp-rec warn"><span class="glyph"></span><span>${recs.who}</span></div>
    <div class="lp-rec ok"><span class="glyph"></span><span>${recs.ok}</span></div>`;
  $('#left-panel').querySelector('.lp-section:nth-of-type(3)').innerHTML = `
    <h4>Activity guidance</h4>
    <div class="lp-rec"><span class="glyph"></span><span>${recs.ex}</span></div>
    <div class="lp-rec"><span class="glyph"></span><span>${recs.win}</span></div>
    <div class="lp-rec"><span class="glyph"></span><span>${recs.mask}</span></div>`;
  $('#lp-conf').textContent = cell.conf;
  $('#lp-traffic').textContent = '+' + Math.max(0, (cell.pm-12)/6).toFixed(1);
  $('#lp-windadj').textContent = (cell.pm > 18 ? '−' : '+') + (Math.abs(Math.sin(cell.i)*1.4)).toFixed(1);
  $('#lp-hwy').textContent = `${Math.round(120 + Math.abs(Math.cos(cell.i*1.7))*900)} m`;
  $('#lp-sources').textContent = (3 + (cell.i % 4)).toString();

  $('#rp-tag').textContent = `Zip ${cell.zip}`;
  $('#rp-name').textContent = `Cell ${cell.c+1}·${cell.r+1} · ${cell.coverage.split('—')[0].trim()}`;
  $('#rp-lat').textContent = cell.lat.toFixed(3);
  $('#rp-lon').textContent = cell.lon.toFixed(3);
  $('#rp-pm').textContent = cell.pm.toFixed(1);
  $('#rp-aqi').textContent = aqiUS(cell.pm);
  $('#fc-cell').textContent = `cell ${cell.c+1}·${cell.r+1} · zip ${cell.zip}`;
}

// ---------- LEFT PANEL TOGGLE ----------
$('#lp-head').addEventListener('click', ()=>{ $('#left-panel').classList.toggle('expanded'); });

// ---------- VIEW SWITCHING ----------
function setView(v){
  const inCity = v === 'city';
  cityGroup.visible = inCity;
  fpGroup.visible = !inCity;
  activeCam = inCity ? cityCam : fpCam;
  canvas.classList.toggle('fp', !inCity);
  $('#fp-hud').style.display = inCity ? 'none' : 'flex';
  $('#exit-btn').style.display = inCity ? 'none' : 'inline-flex';
  $('#right-panel').style.display = inCity ? 'block' : 'none';
  $('#route-btn').style.display = inCity ? 'inline-flex' : 'none';
  const search = $('#zip-search'); if (search) search.style.display = inCity ? 'flex' : 'none';
  tabs.querySelectorAll('button[data-view]').forEach(b=>{
    b.setAttribute('aria-current', b.dataset.view === v ? 'true' : 'false');
  });
  $('#fc-view').textContent = inCity ? 'city overview' : 'street view';
  for (const L of [...labelEls, ...NEIGH_LABELS, ...ZIP_LABELS]) L.el.style.opacity = inCity ? L.el.style.opacity : '0';
  if (!inCity){
    const cell = selectedCell || CELLS[Math.floor(TOTAL/2)];
    const a = aqiFor(cell.pm);
    const info = buildFPParticles(cell.pm);
    $('#fp-cell').textContent = cell.zip;
    $('#fp-pm').textContent = `${cell.pm.toFixed(1)} µg/m³`;
    $('#fp-cat').textContent = a.label;
    $('#fp-dot').style.background = a.color;
    $('#fp-dot').style.boxShadow = `0 0 8px ${a.color}`;
    const perM3 = Math.round(150 + info.severity*5500);
    $('#fp-density').textContent = `~${perM3.toLocaleString()} / m³`;
    scene.background = new THREE.Color(a.color).lerp(new THREE.Color(0x141220), 0.88);
    scene.fog = new THREE.Fog(new THREE.Color(a.color).lerp(new THREE.Color(0x141220), 0.82), 22, 130);
  } else {
    scene.background = SKY_TOP;
    scene.fog = new THREE.Fog(FOG_FAR, 320, 920);
  }
}

tabs.addEventListener('click', e=>{
  const b = e.target.closest('button[data-view]');
  if (b) setView(b.dataset.view);
});
$('#exit-btn').addEventListener('click', ()=> setView('city'));
$('#btn-dropin').addEventListener('click', ()=> setView('street'));
$('#btn-pin').addEventListener('click', ()=>{
  const btn = $('#btn-pin');
  btn.style.borderColor='#ffd166'; btn.style.color='#ffd166';
  setTimeout(()=>{ btn.style.borderColor=''; btn.style.color=''; }, 900);
});
$('#route-btn').addEventListener('click', ()=>{
  const meta = $('#route-btn').querySelector('.meta');
  const orig = meta.textContent;
  meta.textContent = 'coming soon';
  setTimeout(()=>{ meta.textContent = orig; }, 1800);
});

// ---------- ZIP SEARCH ----------
const zipInput = $('#zip-input');
const zipResults = $('#zip-results');
function zipSearch(query){
  query = (query||'').trim();
  if (!query){ zipResults.innerHTML=''; zipResults.style.display='none'; return; }
  const matches = []; const seen = new Set();
  for (const cell of CELLS){
    if (cell.zip.startsWith(query) && !seen.has(cell.zip)){
      seen.add(cell.zip); matches.push(cell);
      if (matches.length>=8) break;
    }
  }
  if (!matches.length){ zipResults.innerHTML='<div class="zr-empty">No matching zip</div>'; zipResults.style.display='block'; return; }
  zipResults.innerHTML = matches.map(cell=>{
    const a = aqiFor(cell.pm);
    return `<div class="zr-item" data-i="${cell.i}">
      <span class="zr-zip">${cell.zip}</span>
      <span class="zr-dot" style="background:${a.color}"></span>
      <span class="zr-cat">${a.label}</span>
      <span class="zr-pm">${cell.pm.toFixed(1)}</span>
    </div>`;
  }).join('');
  zipResults.style.display = 'block';
}
zipInput.addEventListener('input', e=> zipSearch(e.target.value));
zipInput.addEventListener('focus', e=> zipSearch(e.target.value));
zipResults.addEventListener('click', e=>{
  const item = e.target.closest('.zr-item');
  if (!item) return;
  const cell = CELLS[+item.dataset.i];
  selectCell(cell, { pan:true });
  zipResults.style.display='none';
  zipInput.value = cell.zip;
});
document.addEventListener('click', e=>{
  if (!e.target.closest('#zip-search')) zipResults.style.display='none';
});
zipInput.addEventListener('keydown', e=>{
  if (e.key === 'Enter'){ const first = zipResults.querySelector('.zr-item'); if (first) first.click(); }
  if (e.key === 'Escape'){ zipResults.style.display='none'; zipInput.blur(); }
});

// ---------- FP LOOK ----------
let fpYaw=0, fpPitch=0, fpDrag=false, fpDragStart={x:0,y:0,yaw:0,pitch:0};
canvas.addEventListener('mousedown', (e)=>{
  if (activeCam !== fpCam) return;
  fpDrag = true;
  fpDragStart = { x:e.clientX, y:e.clientY, yaw:fpYaw, pitch:fpPitch };
});
window.addEventListener('mouseup', ()=>{ fpDrag = false; });
window.addEventListener('mousemove', (e)=>{
  if (!fpDrag || activeCam !== fpCam) return;
  const dx = e.clientX - fpDragStart.x;
  const dy = e.clientY - fpDragStart.y;
  fpYaw = fpDragStart.yaw - dx*0.0035;
  fpPitch = THREE.MathUtils.clamp(fpDragStart.pitch - dy*0.0035, -0.5, 0.5);
});

// ---------- INITIAL SELECTION ----------
{
  const w = llToWorld(32.78, -96.80);
  let best = CELLS[0], bestD = Infinity;
  for (const cell of CELLS){
    const d = Math.hypot(cell.x - w.x, cell.z - w.z);
    if (d < bestD){ bestD = d; best = cell; }
  }
  selectCell(best);
}

// ---------- ANIMATE ----------
const clock = new THREE.Clock();
let camSwayT = 0;
let selectPulseT = 0;

function tick(){
  const dt = Math.min(0.05, clock.getDelta());
  const t = clock.getElapsedTime();
  camSwayT += dt;
  selectPulseT += dt;

  // City particle drift — wind-aligned + vertical bob + slight wrap
  for (const pts of particleGroups){
    if (!pts) continue;
    const pos = pts.geometry.attributes.position;
    const phase = pts.geometry.attributes.aPhase;
    const seed = pts.geometry.attributes.aSeed;
    const base = pts.userData.basePositions;
    for (let i=0;i<pos.count;i++){
      const i3 = i*3;
      const sd = seed.array[i];
      const ph = phase.array[i] + t*sd*0.4;
      // wind-aligned drift, with phase loop
      const drift = ((t*sd*1.6 + phase.array[i]) % 12) - 6;
      const px = base[i3+0] + WIND.x*drift + Math.sin(ph)*0.6;
      const py = base[i3+1] + Math.sin(ph*1.4)*0.8;
      const pz = base[i3+2] + WIND.y*drift + Math.cos(ph*0.9)*0.6;
      pos.array[i3+0] = px;
      pos.array[i3+1] = py;
      pos.array[i3+2] = pz;
    }
    pos.needsUpdate = true;
  }

  // FP particles drift + recycle
  if (fpParticleSystem && activeCam === fpCam){
    const pos = fpParticleSystem.geometry.attributes.position;
    const phase = fpParticleSystem.geometry.attributes.aPhase;
    for (let i=0;i<pos.count;i++){
      const i3 = i*3;
      const ph = phase.array[i] + t*0.5;
      pos.array[i3+0] += Math.sin(ph)*0.005 + WIND.x*0.02;
      pos.array[i3+1] += Math.cos(ph*0.8)*0.005;
      pos.array[i3+2] += dt*1.2 + WIND.y*0.02;
      if (pos.array[i3+2] > 32) pos.array[i3+2] -= 64;
      if (pos.array[i3+0] > 42) pos.array[i3+0] = -42;
      if (pos.array[i3+0] < -42) pos.array[i3+0] = 42;
      if (pos.array[i3+1] < 0.2) pos.array[i3+1] = 14;
      if (pos.array[i3+1] > 16) pos.array[i3+1] = 0.4;
    }
    pos.needsUpdate = true;
    const target = new THREE.Vector3(
      Math.sin(fpYaw)*Math.cos(fpPitch),
      1.7 + Math.sin(fpPitch),
      -Math.cos(fpYaw)*Math.cos(fpPitch)
    );
    fpCam.lookAt(target.x, target.y, target.z);
  }

  // Subtle camera sway (city)
  if (activeCam === cityCam){
    cityCam.position.y += Math.sin(camSwayT*0.35)*0.02;
    cityCam.position.x += Math.sin(camSwayT*0.21)*0.008;
  }

  // Selection pulse
  const pulse = 0.85 + Math.sin(selectPulseT*2.4)*0.15;
  selectGlowMat.opacity = 0.22 + pulse*0.16;
  selectRingMat.opacity = 0.65 + Math.sin(selectPulseT*2.4)*0.20;

  projectLabels();
  renderer.render(scene, activeCam);
  requestAnimationFrame(tick);
}
tick();

window.__aeria = { setView, selectCell, CELLS, AQI };
