import {
  // RawShaderMaterial,
  MeshNormalMaterial,
  MeshDepthMaterial,
  // GLSL3,
  TextureLoader,
  DoubleSide,
  Vector2,
  Vector3,
  NoBlending,
} from 'three';
// import { ShaderPass } from "../modules/ShaderPass.js";
import metaversefile from 'metaversefile';
const {useMaterials} = metaversefile;
const {WebaverseRawShaderMaterial} = useMaterials();

const baseUrl = import.meta.url.replace(/(\/)[^\/\\]*$/, '$1');

const loader = new TextureLoader();
const blade = loader.load(`${baseUrl}blade.jpg`);

const grassShaderChunks = {
  attributes: `\
  attribute vec3 position;
  attribute vec3 normal;
  attribute vec2 uv;
  attribute vec3 instanceColor;

  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  uniform mat4 modelMatrix;
  uniform mat3 normalMatrix;
  `,
  uniforms: `\
    uniform float scale;
    uniform vec3 cameraTarget;
    uniform vec3 direction2;
    // uniform sampler2D offsetTexture;
    uniform sampler2D offsetTexture2;
    uniform sampler2D quaternionTexture;
    uniform sampler2D quaternionTexture2;
    // uniform sampler2D scaleTexture;
    uniform vec2 curlTSize;

    uniform float time;
    uniform sampler2D curlMap;
    uniform vec3 boulder;
    uniform float size;

    // out vec3 vNormal;
    varying vec2 vUv;
    varying float vDry;
    varying float vLight;
  `,
  functions: `\
  float inverse2(float m) {
    return 1.0 / m;
  }
  
  mat2 inverse2(mat2 m) {
    return mat2(m[1][1],-m[0][1],
               -m[1][0], m[0][0]) / (m[0][0]*m[1][1] - m[0][1]*m[1][0]);
  }
  
  mat3 inverse2(mat3 m) {
    float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
    float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
    float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];
  
    float b01 = a22 * a11 - a12 * a21;
    float b11 = -a22 * a10 + a12 * a20;
    float b21 = a21 * a10 - a11 * a20;
  
    float det = a00 * b01 + a01 * b11 + a02 * b21;
  
    return mat3(b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11),
                b11, (a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10),
                b21, (-a21 * a00 + a01 * a20), (a11 * a00 - a01 * a10)) / det;
  }
  
  mat4 inverse2(mat4 m) {
    float
        a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3],
        a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3],
        a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3],
        a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3],
  
        b00 = a00 * a11 - a01 * a10,
        b01 = a00 * a12 - a02 * a10,
        b02 = a00 * a13 - a03 * a10,
        b03 = a01 * a12 - a02 * a11,
        b04 = a01 * a13 - a03 * a11,
        b05 = a02 * a13 - a03 * a12,
        b06 = a20 * a31 - a21 * a30,
        b07 = a20 * a32 - a22 * a30,
        b08 = a20 * a33 - a23 * a30,
        b09 = a21 * a32 - a22 * a31,
        b10 = a21 * a33 - a23 * a31,
        b11 = a22 * a33 - a23 * a32,
  
        det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  
    return mat4(
        a11 * b11 - a12 * b10 + a13 * b09,
        a02 * b10 - a01 * b11 - a03 * b09,
        a31 * b05 - a32 * b04 + a33 * b03,
        a22 * b04 - a21 * b05 - a23 * b03,
        a12 * b08 - a10 * b11 - a13 * b07,
        a00 * b11 - a02 * b08 + a03 * b07,
        a32 * b02 - a30 * b05 - a33 * b01,
        a20 * b05 - a22 * b02 + a23 * b01,
        a10 * b10 - a11 * b08 + a13 * b06,
        a01 * b08 - a00 * b10 - a03 * b06,
        a30 * b04 - a31 * b02 + a33 * b00,
        a21 * b02 - a20 * b04 - a23 * b00,
        a11 * b07 - a10 * b09 - a12 * b06,
        a00 * b09 - a01 * b07 + a02 * b06,
        a31 * b01 - a30 * b03 - a32 * b00,
        a20 * b03 - a21 * b01 + a22 * b00) / det;
  }
  
  float transpose2(float m) {
    return m;
  }
  
  mat2 transpose2(mat2 m) {
    return mat2(m[0][0], m[1][0],
                m[0][1], m[1][1]);
  }
  
  mat3 transpose2(mat3 m) {
    return mat3(m[0][0], m[1][0], m[2][0],
                m[0][1], m[1][1], m[2][1],
                m[0][2], m[1][2], m[2][2]);
  }
  
  mat4 transpose2(mat4 m) {
    return mat4(m[0][0], m[1][0], m[2][0], m[3][0],
                m[0][1], m[1][1], m[2][1], m[3][1],
                m[0][2], m[1][2], m[2][2], m[3][2],
                m[0][3], m[1][3], m[2][3], m[3][3]);
  }

  vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
  
  float snoise(vec2 v){
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
             -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
    + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
      dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }
  //	Simplex 3D Noise 
  //	by Ian McEwan, Ashima Arts
  //
  vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
  vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
  
  float snoise(vec3 v){ 
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
  
  // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;
  
  // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );
  
    //  x0 = x0 - 0. + 0.0 * C 
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1. + 3.0 * C.xxx;
  
  // Permutations
    i = mod(i, 289.0 ); 
    vec4 p = permute( permute( permute( 
               i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
             + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
  
  // Gradients
  // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0/7.0; // N=7
    vec3  ns = n_ * D.wyz - D.xzx;
  
    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)
  
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
  
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
  
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
  
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
  
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
  
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
  
  //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
  
  // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                  dot(p2,x2), dot(p3,x3) ) );
  }
  
  vec4 colorNoise(vec2 _st, float _zoom, float _speed, float _alphaSpeed){
    vec2 v1 = _st;
    vec2 v2 = _st;
    vec2 v3 = _st;
    float expon = pow(10.0, _zoom*2.0);
    v1 /= 1.0*expon;
    v2 /= 0.62*expon;
    v3 /= 0.83*expon;
    float n = time*_speed;
    float nr = (snoise(vec3(v1, n)) + snoise(vec3(v2, n)) + snoise(vec3(v3, n))) / 6.0;
    n = time * _speed + 1000.0;
    float ng = (snoise(vec3(v1, n)) + snoise(vec3(v2, n)) + snoise(vec3(v3, n))) / 6.0;
    n = time * _speed + 2000.0;
    float nb = (snoise(vec3(v1, n)) + snoise(vec3(v2, n)) + snoise(vec3(v3, n))) / 6.0;
    n = time * _speed * _alphaSpeed + 3000.0;
    float na = (snoise(vec3(v1, n)) + snoise(vec3(v2, n)) + snoise(vec3(v3, n))) / 6.0;
    return vec4(nr,ng,nb,na);
  }

  float inCubic(in float t) {
    return t * t * t;
  }
  
  float outCubic(in float t ) {
    return --t * t * t + 1.;
  }
  
  vec3 applyVectorQuaternion(vec3 vec, vec4 quat) {
    return vec + 2.0 * cross( cross( vec, quat.xyz ) + quat.w * vec, quat.xyz );
  }
  
  mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
  }
  
  vec3 rotateVectorAxisAngle(vec3 v, vec3 axis, float angle) {
    mat4 m = rotationMatrix(axis, angle);
    return (m * vec4(v, 1.0)).xyz;
  }
  
  // this function applies modulo to a vector to keep it within a min/max range
  vec3 modXZ(vec3 minBound, vec3 maxBound, vec3 p) {
    vec2 size = maxBound.xz - minBound.xz;
    vec2 res = mod(p.xz - minBound.xz, size) + minBound.xz;
    return vec3(res.x, p.y, res.y);
  }
  
  mat4 compose(vec3 position, vec4 quaternion, vec3 scale) {
    mat4 te = mat4(1.);
  
    float x = quaternion.x, y = quaternion.y, z = quaternion.z, w = quaternion.w;
    float x2 = x + x,	y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;
  
    float sx = scale.x, sy = scale.y, sz = scale.z;
  
    te[ 0 ][0] = ( 1. - ( yy + zz ) ) * sx;
    te[ 1 ][1] = ( xy + wz ) * sx;
    te[ 2 ][2] = ( xz - wy ) * sx;
    te[ 3 ][3] = 0.;
  
    te[ 1 ][0] = ( xy - wz ) * sy;
    te[ 1 ][1] = ( 1. - ( xx + zz ) ) * sy;
    te[ 1 ][2] = ( yz + wx ) * sy;
    te[ 1 ][3] = 0.;
  
    te[ 2 ][0] = ( xz + wy ) * sz;
    te[ 2 ][1] = ( yz - wx ) * sz;
    te[ 2 ][2] = ( 1. - ( xx + yy ) ) * sz;
    te[ 2 ][3] = 0.;
  
    te[ 3 ][0] = position.x;
    te[ 3 ][1] = position.y;
    te[ 3 ][2] = position.z;
    te[ 3 ][3] = 1.;
  
    return te;
  }
  vec4 getQuaternionFromAxisAngle(vec3 axis, float angle) {
    vec4 q = vec4(0.);
  
    // http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
  
    // assumes axis is normalized
  
    float halfAngle = angle / 2., s = sin(halfAngle);
  
    q.x = axis.x * s;
    q.y = axis.y * s;
    q.z = axis.z * s;
    q.w = cos(halfAngle);
    
    return q;
  }
  vec4 multiplyQuaternions(vec4 a, vec4 b) {
    vec4 q = vec4(0.);
    
    float qax = a.x, qay = a.y, qaz = a.z, qaw = a.w;
    float qbx = b.x, qby = b.y, qbz = b.z, qbw = b.w;
  
    q.x = qax * qbw + qaw * qbx + qay * qbz - qaz * qby;
    q.y = qay * qbw + qaw * qby + qaz * qbx - qax * qbz;
    q.z = qaz * qbw + qaw * qbz + qax * qby - qay * qbx;
    q.w = qaw * qbw - qax * qbx - qay * qby - qaz * qbz;
    
    return q;
  }
  
  const float bladeLength = 0.1;
  const float cover = .25;
  vec2 uvWarp(vec2 uv, int dx, int dy) {
    return mod(uv - 0.25 + vec2(dx, dy) * 0.5, 1.);
  }
  // this function takes a uv coordinate in [0, 1] and returns the distance to the closest side of the square
  float distanceToSide(vec2 uv) {
    /* float d = length(uv - 0.5);
    return 1. - d/sqrt(0.5*0.5 + 0.5*0.5); */
  
    float d = max(abs(uv.x - 0.5), abs(uv.y - 0.5));
    return 1. - d*2.;
  }
  vec4 fourTap4(sampler2D tex, vec2 uv) {
    vec4 sum = vec4(0.);
    float totalWeight = 0.;
    for (int dx=0; dx<2; ++dx) {
      for (int dy=0; dy<2; ++dy) {
        vec2 uv2 = uvWarp(uv, dx, dy);
        float w = distanceToSide(uv2);
        // w = pow(w, 2.);
        sum += texture2D(tex, uv2).rgba * w;
        totalWeight += w;
      }
    }
    return sum / totalWeight;
  }
  vec4 maxTap4(sampler2D tex, vec2 uv) {
    vec4 sum = vec4(0.);
    float totalWeight = 0.;
    for (int dx=0; dx<2; ++dx) {
      for (int dy=0; dy<2; ++dy) {
        vec2 uv2 = uvWarp(uv, dx, dy);
        float w = distanceToSide(uv2);
        // w = pow(w, 2.);
        sum += texture2D(tex, uv2).rgba * w;
        totalWeight += w;
      }
    }
    // return vec4(normalize(sum.rgb), sum.a);
    // sum.a /= totalWeight;
    // sum.a = pow(sum.a, 0.5);
    // sum.a *= 0.7;
    // sum.a *= 1.5;
    // sum.a = min(sum.a, 1.);
    // sum.rgb *= 0.7;
    return sum;
  }
  vec3 fourTap3(sampler2D tex, vec2 uv) {
    vec3 sum = vec3(0.);
    float totalWeight = 0.;
    for (int dx=0; dx<2; ++dx) {
      for (int dy=0; dy<2; ++dy) {
        vec2 uv2 = uvWarp(uv, dx, dy);
        float w = distanceToSide(uv2);
        // w = pow(w, 2.);
        sum += texture2D(tex, uv2).rgb * w;
        totalWeight += w;
      }
    }
    return sum / totalWeight;
  }
  mat4 makeTranslationMatrix(vec3 translation) {
    mat4 m;
    m[0][0] = 1.;
    m[0][1] = 0.;
    m[0][2] = 0.;
    m[0][3] = 0.;
    m[1][0] = 0.;
    m[1][1] = 1.;
    m[1][2] = 0.;
    m[1][3] = 0.;
    m[2][0] = 0.;
    m[2][1] = 0.;
    m[2][2] = 1.;
    m[2][3] = 0.;
    m[3][0] = translation.x;
    m[3][1] = translation.y;
    m[3][2] = translation.z;
    m[3][3] = 1.;
    return m;
  }
  mat4 makeQuaternionMatrix(vec4 quaternion) {
    return compose(vec3(0.), quaternion, vec3(1.));
  }
  mat4 makeScaleMatrix(vec3 scale) {
    mat4 m;
    m[0][0] = scale.x;
    m[0][1] = 0.;
    m[0][2] = 0.;
    m[0][3] = 0.;
    m[1][0] = 0.;
    m[1][1] = scale.y;
    m[1][2] = 0.;
    m[1][3] = 0.;
    m[2][0] = 0.;
    m[2][1] = 0.;
    m[2][2] = scale.z;
    m[2][3] = 0.;
    m[3][0] = 0.;
    m[3][1] = 0.;
    m[3][2] = 0.;
    m[3][3] = 1.;
    return m;
  }`,
  main: `\

  vec3 offset = vec3(instanceColor.y, 0., instanceColor.z);
  vec2 curlUv = instanceColor.yz / size + 0.5;
  
  const float offsetRange = 2.;
  const float rangeWidth = offsetRange * 2.;
  vec3 ct = cameraTarget/scale;
  vec3 minRange = vec3(ct.x - offsetRange, 0., ct.z - offsetRange);
  vec3 maxRange = vec3(ct.x + offsetRange, 0., ct.z + offsetRange);
  offset = modXZ(
    minRange,
    maxRange,
    offset
  );

  vec3 positionV = texture2D(offsetTexture2, curlUv).rgb;
  vec4 quaternionV1 = texture2D(quaternionTexture, curlUv);
  vec4 axisAngleV = texture2D(quaternionTexture2, curlUv);
  vec4 quaternionV2 = getQuaternionFromAxisAngle(axisAngleV.rgb, axisAngleV.a);
  vec4 quaternionV = multiplyQuaternions(quaternionV1, quaternionV2);
  vec3 scaleV = vec3(1., 1., bladeLength);

  float id = instanceColor.x;
  // vec2 curlTSize = vec2(textureSize(curlMap, 0));
  vec2 curlUv2 = vec2(offset.x, offset.z);
  vec4 curlV = colorNoise(curlUv2 * 400. + id * 0.0002, 1., 4., 0.5);
  curlV.rgb *= 30.;
  // curlV.rb *= 0.2;
  // curlV.g -= 10.;
  curlV.a += 0.5;
  // curlV.a *= 1.25;

  // base position
  vUv = vec2(uv.x, 1.-uv.y);
  vec3 base = vec3(position.xy, 0.) + offset;
  vec3 dBoulder = mod(boulder-base + rangeWidth / 2., rangeWidth) - rangeWidth / 2.;
  vLight = (1./length(dBoulder))/10.;
  vLight = pow(vLight, 2.);
  /* if(length(dBoulder)>cover) {
    dBoulder = vec3(0.);
  } */

  // curl
  vec3 n = curlV.xyz;
  float h = (1.+ curlV.a);
  float l = length(dBoulder) > 0. ? (length(dBoulder)/cover) : 0.;
  vec3 pNormal = (transpose2(inverse2(modelMatrix)) * vec4(normalize(vec3(.01 * n.xy, 1.)), 1.)).xyz;
  vec3 target = normalize(position + pNormal) * h;
  // vNormal = normalMatrix * normal;
  vec3 transformed = position;
  float f = inCubic(position.z);
  transformed = mix(transformed, target, f);
  // p = mix(p, p - dBoulder * l, f);
  // p *= length(dBoulder);

  vDry = curlV.a;

  // p = rotateVectorAxisAngle(p, vec3(0, 0., 1.), PI/2. + atan(-direction2.z, -direction2.x));
  mat4 instanceMatrix = makeScaleMatrix(vec3(scale)) *
    makeTranslationMatrix(offset) *
    compose(positionV, quaternionV, scaleV) *
    rotationMatrix(vec3(0, 0., 1.), PI/2. + atan(-direction2.z, -direction2.x));
  `
};

const vertexShader = `precision highp float;
#define USE_INSTANCING
#ifndef PI
  #define PI 3.14159265358979323846
#endif
${grassShaderChunks.attributes}
${grassShaderChunks.uniforms}
${grassShaderChunks.functions}

void main() {
  ${grassShaderChunks.main}

  vec4 mvPosition = vec4( transformed, 1.0 );
  #ifdef USE_INSTANCING
    mvPosition = instanceMatrix * mvPosition;
  #endif
  mvPosition = modelViewMatrix * mvPosition;
  gl_Position = projectionMatrix * mvPosition;
}`;

const fragmentShader = `precision highp float;

varying vec2 vUv;
varying float vDry;
varying float vLight;

uniform sampler2D blade;

// out vec4 fragColor;

void main() {
  vec4 c = texture2D(blade, vUv);
  if (c.r < .5) {
    discard;
  }
  vec3 color1 = vec3(75., 112., 34.) / 255.;
  vec3 color2 = vec3(93., 128., 47.) / 255.;
  vec3 color3 = vec3(102., 146., 44.)/ 255.;
  vec3 color4 = vec3(216., 255., 147.)/ 255.;

  vec3 color = mix(mix(color1, color2, vUv.y), color3, vDry);
  color -= 0.05;
  color += vLight;
  // color = mix(color, color4, vLight);
  gl_FragColor = vec4(color * vUv.y, 1.);
}`;

const _addGrassMaterialUniforms = uniforms => {
  uniforms.scale = { value: 1 };
  // uniforms.curlMap = { value: null };
  uniforms.boulder = { value: new Vector3() };
  uniforms.size = { value: 0 };
  uniforms.time = { value: 0 };
  uniforms.persistence = { value: 1 };
  uniforms.blade = { value: blade };
  uniforms.cameraTarget = { value: new Vector3() };
  uniforms.direction2 = { value: new Vector3() };
  // uniforms.offsetTexture = { value: null };
  uniforms.offsetTexture2 = { value: null };
  uniforms.quaternionTexture = { value: null };
  uniforms.quaternionTexture2 = { value: null };
  // uniforms.scaleTexturev { value: null };
  return uniforms;
};

class GrassMaterial extends WebaverseRawShaderMaterial {
  constructor(options) {
    const uniforms = _addGrassMaterialUniforms({});
    super({
      vertexShader,
      fragmentShader,
      // glslVersion: GLSL3,
      ...options,
      uniforms,
      side: DoubleSide,
      transparent: true,
    });
  }
}

class GrassDepthMaterial extends MeshNormalMaterial {
  constructor(options = {}) {
    super(options);
    this.blending = NoBlending;
    this.uniforms = null;
  }
  onBeforeCompile(parameters) {
    parameters.uniforms = _addGrassMaterialUniforms(parameters.uniforms);
    // console.log('set uniforms', parameters.uniforms);
    this.uniforms = parameters.uniforms;
    const preMain = `\
    ${grassShaderChunks.uniforms}
    ${grassShaderChunks.functions}
    `;
    const postMain = `\
      ${grassShaderChunks.main}
    `;

    parameters.vertexShader = parameters.vertexShader.replace('void main() {\n', preMain + 'void main() {\n' + postMain);
    parameters.vertexShader = parameters.vertexShader.replace('#include <begin_vertex>\n', `\
      // vec3 transformed = vec3( position );
    `);
    parameters.vertexShader = parameters.vertexShader.replace('#include <defaultnormal_vertex>\n', `\
    vec3 transformedNormal = objectNormal;
    #ifdef USE_INSTANCING
      // this is in lieu of a per-instance normal-matrix
      // shear transforms in the instance matrix are not supported

      mat4 instanceMatrix2 = makeQuaternionMatrix(quaternionV) *
        rotationMatrix(vec3(0, 0., 1.), PI/2. + atan(-direction2.z, -direction2.x));

      mat3 m = transpose2(inverse2(mat3(instanceMatrix2)));
      // mat3 m = mat3( instanceMatrix );
      // transformedNormal /= vec3( dot( m[ 0 ], m[ 0 ] ), dot( m[ 1 ], m[ 1 ] ), dot( m[ 2 ], m[ 2 ] ) );
      transformedNormal = m * transformedNormal;
    #endif
    transformedNormal = normalMatrix * transformedNormal;
    #ifdef FLIP_SIDED
      transformedNormal = - transformedNormal;
    #endif
    #ifdef USE_TANGENT
      vec3 transformedTangent = ( modelViewMatrix * vec4( objectTangent, 0.0 ) ).xyz;
      #ifdef FLIP_SIDED
        transformedTangent = - transformedTangent;
      #endif
    #endif
    `);

    parameters.fragmentShader = parameters.fragmentShader.replace('void main() {\n', `\
      uniform sampler2D blade;
      varying vec2 vUv;
    ` + 'void main() {\n' + `\
      vec4 c = texture2D(blade, vUv);
      if (c.r < .5) {
        discard;
      }
    `);

    // console.log('got normal map shader', parameters.vertexShader, parameters.fragmentShader);
  }
}

export { GrassMaterial, GrassDepthMaterial };
