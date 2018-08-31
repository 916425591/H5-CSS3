/*
	NOTES and ACKNOWLEDGMENTS:
	Procedural terrain with waterlike surface, rocks and a low quality seagull.

	The rocks were influenced by:
    'Nightfall' by vgs https://www.shadertoy.com/view/MlfXWH
	The color of the rocks was taken (with slight modification) from
	'Tiny Planet' by valentingalea https://www.shadertoy.com/view/ldyXRw

	Implementations that I took directly from Inigo's articles:

	Altitude based fog (took the funciton as is).
	http://iquilezles.org/www/articles/fog/fog.htm

	Implemented penumbras as explained here:
	http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm

	Used signed distance modeling with functions:
	http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

	I guess that the hash functions are also originated from Inigo's work.

	Also thanks to VGS for explanation on smooth min and one line rotations:
	http://homepages.dcc.ufmg.br/~vgs/blog/simple-code-rotation/

	Reflection and refraction were taken from scratchpixel.
	The refracted objects look dimmer with deps dependent on distance measured. Used
	exponential attenuation.

	The scattering was taken from scratchpixel as well, however with some changes:
	Changed some of the defaults: (BetaR, atmosphere height to be 100km - same as earth,
	By default the step is exponential and not uniform. This creates smoother transitions and a
	better sunset effect, however it introduces more error at the higher altitudes due to lack
	of samples there. So for the upper half, there is only one sample which is taken at the point
	.75 of the atmosphere. It is possible to improve it by picking the point with an average density
	for each range (and for .5 - 1. of the atmosphere specifically). Clearly this point should
	be closer to the lower bound of the range.

	commenting EXPSTEP will use uniform step.
	precomputed values for BetaR:
	use RED_SUNSET for aqua like sky color and red sunset.
 	use YELLOW_SUNSET for blue sky and yellow sunset.

	TODO:
	Use analytic raytracing for the sea surface.
	Better camera movements.
	Maybe clouds.
*/
#define PI 3.14159265359

#define HGN 9999.9
#define FAR 400.
#define EPS (2.0/iResolution.x)
#define WATER_VIS_COEFF .7
#define PEAK 14.
#define EXP1 2.718282

#define IGT iTime
#define EARTH_ROT_TIME IGT/13.

#define CAMERA_MOVE -5.*IGT

#define EARTH_RADIUS 6370e3
#define ORBIT_RADIUS 6470e3

#define VIEW_RAY_SAMPLES 12
#define SUN_RAY_SAMPLES 6

#define EXPSTEP
#define EXP16 -0.692885385572419
#define EXP8 -0.69102321753708

#define HR 7994. // Rayleigh Height
#define HM 1200. // Mie Height

#define YELLOW_SUNSET vec3(5.8e-6, 10.5e-6, 9.1e-5)
#define RED_SUNSET vec3(5.8e-6, 80.5e-6, 9.1e-5)

vec3 BetaR = vec3(5.8e-6, 10.5e-6, 9.1e-5);
const vec3 BetaM = vec3(21e-6);

vec2 solveQuadratic(in float a, in float b, in float c) { // the second value is 0 if no solution exists and 1 if found solution.
    float discr = b*b - 4.*a*c;
    vec2 res = vec2(0.0, 0.0);
    float eps = 0.001;

    if (discr > 0.0) {
        float sq_discr = sqrt(discr);
        float q = -.5*(b+sign(b)*sq_discr);
        float t1 = q/a;
        float t2 = c/q;

        // the solution is the minimal positive t.

        res.x = t1 * t2 > 0. ? min(t1, t2) : max(t1, t2);
        res.y = step(eps, max(t1, t2));
    }

    return res;
}

struct Ray {
    vec3 o;
    vec3 d;
};

struct ODist {
    float t;
    int idx;
    float eps;
    float pmb;
};

float raySphereIntersect(in Ray ray, in vec4 sphere) { // vec4(ox, oy, oz, radius);
    vec3 o = ray.o - sphere.xyz;
    vec3 d = ray.d;
    float t = -1.0;

    float a = dot(d, d);
    float b = 2.*dot(o, d);
    float c = dot(o, o) - sphere.w*sphere.w;

    vec2 res = solveQuadratic(a, b, c);

    if (res.y > 0.5) {
        t = res.x;
    }

    return t;
}

float earthOrbitDist(in Ray ray) {
    return raySphereIntersect(ray, vec4(0., 0., 0., ORBIT_RADIUS));
}

#define MAX_RAY_MARCH 300

float random (vec2 st) {
    return fract(sin(dot(st.xy,
        vec2(12.9898,78.233)))*
        43758.5453123);
}

float random(vec3 t) {
    return fract(sin(dot(t.xyz,
        vec3(12.9898,78.233, 2.23435)))*
        43758.5453123);
}

vec2 rot2d(in vec2 u, in float a) {
    return u.xy*cos(a) + vec2(-u.y, u.x)*sin(a);
}

float value_noise(in vec3 p) {
    vec3 ip = floor(p);
    vec3 fp = fract(p);

    float q = random(ip);
    float w = random(ip + vec3(1.,0.,0.));
    float e = random(ip + vec3(0.,1.,0.));
    float r = random(ip + vec3(1.,1.,0.));

    float a = random(ip + vec3(0.,0.,1.));
    float s = random(ip + vec3(1.,0.,1.));
    float d = random(ip + vec3(0.,1.,1.));
    float f = random(ip + vec3(1.,1.,1.));

    vec3 u = 3.*fp*fp - 2.*fp*fp*fp;

    float v1 = mix(mix(q,w,u.x),
        mix(e,r,u.x), u.y);
    float v2 = mix(mix(a,s,u.x),
        mix(d,f,u.x), u.y);
    float v = mix(v1, v2, u.z);
    return v;
}

vec3 value_noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f*f*f*(f*(f*6.0 - 15.0) + 10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0); //iq

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = a - b - c + d;

    vec3 nres;

    nres.x = mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    nres.yz = vec2(k1+k3*u.y, k2+k3*u.x)*du;

    return nres;
}

#define OCTAVES 10

float waterFbm(in vec3 p, in int octaves)  {
    float f = 0., af = 0.;
    /*float f = .5*value_noise(p);
    f += .25*value_noise(2.*p);
    f += .125*value_noise(4.*p);
    f += .0625*value_noise(8.*p);
    return f;*/

    for (int i = 0; i < 4; i++) {
        if (i >= octaves) break;
        af = pow(2., float(i));
        f += (.5/af)*value_noise(af*p);
    }

    return f;
}

vec3 fbm(in vec2 grid, in int octaves) {
    // Initial values
    float v = 0.0, a = PEAK, f = 2.101;
    mat2 m = mat2(0.6, -.8, .8, .6);
    mat2 cm = mat2(1., 0., 0., 1.);
    vec2 d = vec2(0., 0.);
    // Loop of octaves
    a *= sqrt(abs(value_noise(grid*.3).x));

    for (int i = 0; i < OCTAVES; i++) {
        if (i >= octaves) break;
        vec3 ns = value_noise(grid);
        v += (a * ns.x)/(.78+dot(ns.yz, ns.yz));
        d += (a * cm * ns.yz);
        grid = m*grid*f;
        cm = m*cm;
        a *= .49;
    }
    return vec3(v, d);
}

#define NUM_GEOMS 3

#define c_beach vec3(.153, .172, .121)
#define c_rock  vec3(.080, .050, .030)
#define c_snow  vec3(0.805671, 0.805671, 0.805671)

vec3 origin;

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5*(a-b)/k, 0.0, 1.0);
    return mix(a, b, h) - k*h*(1.0-h);
}

float getMGeomL(in vec3 p) {
    return p.y - fbm(p.xz*.05, 2).x;
}

float getMGeom(in vec3 p) {
    return p.y - fbm(p.xz*.05, 4).x;
}

float getMGeomH(in vec3 p) {
    float d = length(p - origin);
    //int octaves = 7+int(log2(1.-d/FAR));
    int octaves = int(7.*(1. - .5*d/FAR)+1.);
    return p.y - fbm(p.xz*.05, octaves).x;
}

vec4 getMMtl(in vec3 p, in vec3 N) {
    vec3 col;
    float s = smoothstep(15., 19., p.y);
    col = mix(c_rock, c_snow, s*sqrt(abs(N.y)));
    s = smoothstep(12., 13., p.y);
    col += mix(c_beach, c_rock, s);

    return vec4(col, 10.);
}

struct Wave {
    float a;
    float L;
    float v;
    vec2  dir;
    float k;
};

float wave(in vec2 st, in Wave w) {
    float res;

    float ampl = w.a;
    float L = w.L;
    float freq = 2./L;
    vec2  dir = normalize(w.dir);
    float speed = w.v;
    float phi = speed * 2./L;

    res = 2.*ampl * pow(.5*(1.+sin(dot(dir,st)*freq + phi)), w.k);
    return res;
}

vec3 gwave(in vec2 st, in Wave w) {
    vec3 res;

    float ampl = w.a;
    float L = w.L;
    float freq = 2./L;
    vec2  dir = normalize(w.dir);
    float speed = w.v;
    float phi = speed * 2./L;
    float q = 1./(freq*ampl);

    res.x = q*ampl + dir.x*cos(dot(dir, st)*freq + phi);
    res.y = q*ampl + dir.y*cos(dot(dir, st)*freq + phi);
    res.z = ampl * sin(dot(dir,st)*freq + phi);
    return res;
}

float getWGeom(in vec3 p) {
    Wave waves[3];
    waves[0] = Wave(1.2, 15., 3.5 + IGT*4., vec2(1., 1.), .7);
    waves[1] = Wave(1.1, 12.6, 2.5 + IGT, vec2(-1., -2.), .9);
    waves[2] = Wave(.4, 4., 2. + IGT*2., vec2(1., -.6), .9);
    waves[2] = Wave(.4, 4.2, 1. + IGT*2., vec2(-1., -.6), .9);

    /*for (int i = 0; i < 3; i++) {
        p.y += wave(p.xz, waves[i]);
    }*/
    for (int i = 0; i < 3; i++) {
        vec3 q = gwave(p.xz, waves[i]);
        p.xzy = p.xzy + q.xyz;
    }

    return p.y -12.;
}

float getWGeomH(in vec3 p) {
    return getWGeom(p);

    float d = length(p - origin);
    //int octaves = 4+int(log2(1.-d/FAR));
    int octaves = int(4.*(1. - .8*d/FAR)+1.);
    return p.y -12. -waterFbm(vec3(p.xz*.1, IGT/4.), octaves);
}

vec3 wingTrf(in vec3 p) {
    vec3 q = vec3(p.xy, p.z-2.6 + 2.*pow(abs(cos(p.y/4.)), 4.));
    q.z += cos(IGT*8. + PI/2.)*.04*q.y*q.y;
    return q;
}

float birdWings(in vec3 p) {
    vec3 q = wingTrf(p);

    q.y *= .3;
    q.x += .6*cos(2.5*q.y);

    float f1 = max(length(q.xy) - 3.5, abs(q.z) - .01);

    vec3 q2 = wingTrf(p);
    q2.y *= .75;
    float cq2y = cos(q2.y);
    q2.x += cq2y*cq2y*cq2y*.343;
    float f2 = max(length(q2.xy - vec2(-12., 0.)) - 12.5, abs(q2.z) - .01);

    float f = max(f1, f2);

    return f;
}

float birdBody(in vec3 p) {
    vec3 q = p;

    float l = length(q.yz);
    float f = max(l - (1.-.08*q.x*q.x - .3*q.x) -.3, abs(q.x) - 7.);
    return f;
}

#define BR -180.

float getSBird(in vec3 p) {
    p -= vec3(-BR*sin(.5*IGT) + origin.x, 21., BR*cos(.5*IGT) + origin.z);
    p.yz = rot2d(p.yz, PI/3.);
    return smin(birdBody(p), birdWings(p), .2);
}

vec4 seagullCol(in vec3 p) {

    p -= vec3(-BR*sin(.5*IGT) + origin.x, 21., BR*cos(.5*IGT) + origin.z);
    p.yz = rot2d(p.yz, PI/3.);
    vec3 col;
    float wgs = birdWings(p);
    float bdy = birdBody(p);

    col = smoothstep(0., .3, .3-abs(wgs))*mix(vec3(1.,1.,1.), vec3(0.1, 0.1, 0.1), step(7.5, abs(p.y)));
    col += smoothstep(0., .3, .3-abs(bdy))*vec3(1., 1., 1.);

    return vec4(col, 6.2);
}

ODist intersect(in vec3 p, in int excl_idx) {
    ODist gi[NUM_GEOMS];
    gi[0] = ODist(-1., 1, getMGeom(p), 1.);
    gi[1] = ODist(-1., 2, getWGeom(p), 1.);
    gi[2] = ODist(-1., 3, getSBird(p), 1.);

    ODist g = ODist(HGN, 0, HGN, 1.);

    for (int i = 0; i < NUM_GEOMS; i++) {
        if (gi[i].eps < g.eps && excl_idx != gi[i].idx) {
            g = gi[i];
        }
    }

    return g;
}

vec3 calcNormal(in vec3 p, int geomId) {
    vec3 e = vec3(0.001, 0.0, 0.0);
    vec3 n;

    if (geomId == 1) {
        //vec2 n1 = fbm(p.xz*.05, 4).yz;
        //n = normalize(vec3(n1.x, 0.001*2., n1.y));

        n.x = getMGeomH(p+e.xyy) - getMGeomH(p-e.xyy);
        n.y = 2.*e.x;
        n.z = getMGeomH(p+e.yyx) - getMGeomH(p-e.yyx);
    } else if (geomId == 2) {
        //n = vec3(0., 1., 0.);

        n.x = getWGeomH(p+e.xyy) - getWGeomH(p-e.xyy);
        n.y = 2.*e.x;
        n.z = getWGeomH(p+e.yyx) - getWGeomH(p-e.yyx);

    } else if (geomId == 3) {
        n.x = getSBird(p+e.xyy) - getSBird(p-e.xyy);
        n.y = getSBird(p+e.yxy) - getSBird(p-e.yxy);
        n.z = getSBird(p+e.yyx) - getSBird(p-e.yyx);
    }

    return normalize(n);
}

ODist trace(in Ray ray, in int excl_idx) {
    ODist res = ODist(-1.0, 0, HGN, 1.);
    res.t = -1.;
    res.idx = 0;
    float pmb = 10.;
    float t = 0.0;
    float up = step(0.01, ray.d.y);

    for (int it = 0; it < MAX_RAY_MARCH; it++) {
        vec3 p = ray.o + t*ray.d;
        if (up*step(2.*PEAK, p.y) > .5) {
            break;
        }
        res = intersect(ray.o + t*ray.d, excl_idx);
        if (res.eps < 0.001*t) {
            res.t = t;
            res.pmb = 0.;
            return res;
        }

        pmb = min(pmb, 10.*res.eps/(t + 0.001));
        //t += (.25+exp(5.*t/FAR-5.))*res.eps;
        t += .25*res.eps;

        if (t > FAR) {
            break;
        }
    }

    return ODist(t, 0, res.eps, pmb);
}

#define NUM_LIGHTS 1
#define DIR_LIGHT 1
#define SPH_LIGHT 2

struct Light {
    int type;
    vec4 col;
    vec3 pos;
    vec3 dir;
};

float fresnell(in vec3 I, in vec3 N, in float ior) {
    float sint1 = length(cross(I, N));
    float sint2 = sint1*ior;

    if (sint2 > 1.) {
        return 1.;
    }

    float cost1 = cos(asin(sint1));
    float cost2 = cos(asin(sint2));

    float fpar = (ior*cost1 - cost2)/(ior*cost1 + cost2);
    float fperp = (cost2 - ior*cost1)/(cost2 + ior*cost1);

    float fresnell = (fpar*fpar + fperp*fperp)*.5;
    return fresnell;
}

vec3 scattering(in Ray ray, in vec3 sun_dir) {
    float to = earthOrbitDist(ray);
    vec3 col = vec3(.5);
    float ds = to/float(VIEW_RAY_SAMPLES);
    float mu = dot(ray.d, sun_dir);
    float phaseR = 3./(16.*PI)*(1.+mu*mu);
    float g = 0.80;
    float denom = 1. + g * g - 2. * g * mu;
    float phaseM = 3. / (8. * PI) * ((1. - g * g) * (1. + mu * mu)) / ((2. + g * g) * denom*denom/sqrt(denom));
    float opR = 0.0;
    float opM = 0.0;
    vec3 sumR = vec3(0.0);
    vec3 sumM = vec3(0.0);

    for (int i = 0; i < VIEW_RAY_SAMPLES; i++) {
    #ifdef EXPSTEP
        ds = exp(EXP16*float(VIEW_RAY_SAMPLES - i))*to;
        #endif
        vec3 x = ray.o + (ds*float(i) + ds*.5)*ray.d;
        float height = length(x)-EARTH_RADIUS;
        float hr = exp(-height/HR)*ds;
        float hm = exp(-height/HM)*ds;
        float opSunR = 0.0;
        float opSunM = 0.0;
        vec3 attenuation = vec3(0.0);
        opR += hr;
        opM += hm;

        Ray sun_ray = Ray(x, sun_dir);
        float tsun = earthOrbitDist(sun_ray);
        float dst = tsun/float(SUN_RAY_SAMPLES);

        for (int j = 0; j < SUN_RAY_SAMPLES; j++) {
        #ifdef EXPSTEP
            dst = exp(EXP8*float(SUN_RAY_SAMPLES - i))*tsun;
            #endif
            vec3 xt = sun_ray.o + (dst*float(j)+dst*.5)*sun_ray.d;
            float htt = length(xt) - EARTH_RADIUS;
            opSunR += exp(-htt/HR)*dst;
            opSunM += exp(-htt/HM)*dst;
        }

        attenuation = exp(-BetaR*(opR + opSunR) -BetaM*(opM + opSunM)*1.1);
        sumR += attenuation*hr;
        sumM += attenuation*hm;
    }

    col = (sumR*phaseR*BetaR*(1.+max(sun_dir.y, 0.)) + sumM*phaseM*BetaM)*15.;
    return col;
}

vec3 render(in Ray ray, in ODist tgeom, in Light l, bool shadow) {
    vec3 col = vec3(0., 0., 0.);
    vec3 p = ray.o + tgeom.t*ray.d;
    vec3 n = calcNormal(p, tgeom.idx);

    vec3 ldir = l.dir;
    //vec3 ldir = normalize(l.pos - p); // spherical light
    vec3 vreflect; // = ldir - 2.*dot(n, ldir)*n;
    vreflect = reflect(ldir, n);

    if (tgeom.idx == 1) {
        vec4 mtl = getMMtl(p, n);
        vec3 diff = mtl.xyz*l.col.xyz/(PI)*max(0.0, dot(n, ldir.xyz))*l.col.w;
        vec3 spec = l.col.xyz*pow(max(0.0, dot(vreflect, ray.d)), mtl.w)*l.col.w/mtl.w;
        col = diff + .1*spec;
    }  else if (tgeom.idx == 3) {
        vec4 mtl = seagullCol(p);
        vec3 diff = mtl.xyz*l.col.xyz/(PI)*max(0.0, dot(n, ldir.xyz))*l.col.w;
        col = diff*(.2 + step(0., ldir.y));
    }  else {
        Ray sray = Ray(ray.o + vec3(0., EARTH_RADIUS, 0.), ray.d);
        col = scattering(sray, ldir);
    }

    if (tgeom.idx != 0 && shadow) {
        vec3 vpos = p+normalize(ldir)*.1;
        Ray shRay = Ray(p, ldir);
        float pmb = float(trace(shRay, -1).pmb);
        col *= min(1., .5+pmb);
    }

    return col;
}

vec3 reflection(in vec3 I, in vec3 N, in vec3 p, in Light l, in int gindex) {
    vec3 vreflect = reflect(I, N);
    Ray reflRay = Ray(p, vreflect);
    ODist tgeom2 = trace(reflRay, gindex);
    return render(reflRay, tgeom2, l, false);
}

vec4 refraction(in vec3 I, in vec3 N, in vec3 p, in Light l, in float ior, in int gindex) {
    vec3 vrefract = refract(I, N, ior);
    Ray refrRay = Ray(p, vrefract);
    ODist tgeom = trace(refrRay, gindex);
    return vec4(render(refrRay, tgeom, l, false), tgeom.t);
}

vec3 applyFog(in vec3 rgb, in float distance, in Ray ray) {
    float b = .3 + .03*sin(IGT/4.234);
    float c = 1.;
    float fogAmount = c * exp(-ray.o.y*b) * (1.0-exp( -distance*ray.d.y*b ))/ray.d.y;
    vec3  fogColor  = vec3(0.5,0.6,0.9);
    return mix( rgb, fogColor, fogAmount );
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 st = fragCoord.xy / iResolution.xy;
    st.x = 1. - st.x;
    //st.x *= iResolution.x/iResolution.y;
    vec3 col = vec3(0.0);
    Ray ray;
    ray.o = vec3(0, 30. + 2.*sin(IGT/4.), 2. CAMERA_MOVE);
    origin = ray.o;

    ray.d = normalize(vec3(-1. + 2.*st.xy, -1.));

    // First iteration
    ODist tgeom = trace(ray, 0);
    vec3 ldir = normalize(vec3(-2., 3., -5.));
    float rot_cf = 1.;
    if (iMouse.y > 0.) {
        rot_cf = pow(abs(2.*iMouse.y/iResolution.y), 4.);
    } if (iMouse.x >0.) {
        BetaR = mix(YELLOW_SUNSET, RED_SUNSET, iMouse.x/iResolution.x);
    }
    ldir.yz = rot2d(ldir.yz, -EARTH_ROT_TIME*rot_cf);
    Light l = Light(2, vec4(1., 1., 1., 4.4), vec3(0., 10., -10.), ldir);
    if (tgeom.idx != 2) {
        col = render(ray, tgeom, l, true);
    } else if (tgeom.idx == 2) {
        float ior = 1./1.3;
        vec3 p = ray.o + ray.d*tgeom.t;
        vec3 n = calcNormal(p, tgeom.idx);
        vec3 vreflect = reflect(ray.d, n);
        float frs = fresnell(ray.d, n, ior);
        vec3 col1 = reflection(ray.d, n, p, l, tgeom.idx);
        vec4 refr = refraction(ray.d, n, p, l, ior, tgeom.idx);
        // darken refraction based on ray length.
        col = .85*(frs*col1 + .3*(1.-frs)*refr.xyz*exp(-refr.w*WATER_VIS_COEFF));
    }

    col = applyFog(col, tgeom.t, ray);

    fragColor = vec4(col, 1.0);
}