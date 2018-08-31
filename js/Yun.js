/**
 * Written by Gerard Geer.
 * License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
 *
 * Version 1.0:	Initial release.
 * Version 1.1:	Added a couple quality defines. CALC_OCCLUSION specifies to calculate
 * 				ambient occlusion, which is about 5 calls to dist(), each of which has
 * 				about 10 texture samples.
 * 				HQ_RENDER does a whole lot. In its absence, the FBMs have one fewer
 * 				octave, the rock loses a middle octave of noise, the sand is no longer
 * 				image-reflective, and the field as in the ocean reflections is textured
 * 				only with the sand texture. This cuts out a whole lot of texture calls,
 * 				and a deep-in-there branch.
 * 				These defines are commented by default. Enable them to see the whole
 * 				picture.
 */

// Render quality defines.
// #define CALC_OCCLUSION
// #define HQ_RENDER

// Occlusion attenuation factor.
#define OCC_FACTOR 1.5

// Light and reflection penumbra factors.
#define PENUMBRA_FACTOR 50.0
// Oren-Nayar material reflectance coefficient.
#define MAT_REFLECTANCE 4.0
// Main marching steps.
#define V_STEPS 100
// Reflection marching steps.
#define R_STEPS 30
// Shadow marching steps.
#define S_STEPS 30
// Maximum successful marching distance.
#define EPSILON .025
// Max ray depth.
#define MAX_DEPTH 150.0

// How fast do you want time to go?
#define TIME_WARP .3

// Object constants.
#define SAND_NOISE_TEX iChannel1
#define SAND_BUMP_HEIGHT .6
#define OCEAN_NOISE_TEX iChannel1
#define OCEAN_HEIGHT -.25
#define OCEAN_AMPLITUDE .45
#define OCEAN_SPEED .35
#define WALL_BUMP_HEIGHT 7.5
#define WALL_DIFF_TEX iChannel2
#define RAND_NOISE_TEX iChannel0
const vec2 ROCK_DIM = vec2(4.0,1.75);

// Object IDs for shading.
#define ID_ROCK 1.0
#define ID_SAND 128.0
#define ID_OCEAN 256.0

// Environment and color constants.
const vec3 UP = vec3(0.0, 1.0, 0.0);						// An up vector.
const vec3 PLANET_ROT = vec3(0.0,0.0,1.0);					// The axis on which the planet rotates.
const vec3 SUN_DIR = vec3(0., 0.995037, 0.0995037);			// The starting direction of the sun.
const vec3 SKY_COLOR_A = vec3(0.53,0.81,0.92);			    // Sky color.
const vec3 SKY_COLOR_B = vec3(0.23,0.34,0.85);				// High angle sky.
const vec3 SUN_COLOR_A = vec3(4.0);							// Noontime sun color.
const vec3 SUN_COLOR_B = vec3(2.0, .66, 0.06);				// Evening sun color.
const vec3 GROUND_COLOR = vec3(.6, .5, .32);				// Average ground color.
const vec3 SAND_COLOR = vec3(.4, .35, .22);					// The color of the sand.
const vec3 ROCK_COLOR_A = vec3(.4,.3,.15);					// One rock color.
const vec3 ROCK_COLOR_B = vec3(.4,.2,.15);					// Another rock color.
#define NIGHT_BRIGHTNESS .1

// Math constants.
const mat2 ROT45 = mat2(-.710, -.710, .710, -.710);
const mat2 ROT90 = mat2(0.,1.,-1.,0.);
const vec2 RAND_SEED = vec2(12.9898,78.233);
#define PI   3.141593
#define PI4 12.56637

// Camera parameters.
const vec3 CAM_LOOK_AT = vec3(0.0, 2.0, 0.0);
const float CAM_DIST_AWAY = 50.25; // How far away the camera is from CAM_LOOK_AT.
const float CAM_ZOOM = 1.0;

/*
	Olinde Rodrigues' vector rotation formula for rotating a vector <a>
	around a vector <b> <t> radians.
*/
vec3 rodRot( in vec3 a, in vec3 b, in float t )
{
    // Straight from wikipedia.
    return normalize(a*cos(t) + cross(b, a)*sin(t) + b*dot(b,a)*(1.0-cos(t)));
}

/*
	My goofy redefinition-of-basis-based camera.
*/
void camera(in vec2 uv, in vec3 cp, in vec3 cd, in vec3 up, in float f, out vec3 ro, out vec3 rd)
{
    ro = cp;
    rd = normalize((cp + cd*f + cross(cd, up)*uv.x + up*uv.y)-ro);
}

/*
	Returns a coefficient for a shutter fade.
*/
float shutterfade( in float s, in float e, in float t, in float duration )
{
    return min( smoothstep(s, s+duration, t), smoothstep(e, e-duration, t) );
}

/*
	A decoration function to get between you and and camera().
	Nevermind that it does fancy animation; death to interface degraders!
*/
void animateCam(in vec2 uv, in float t, out vec3 p, out vec3 d, out vec3 e, out float s )
{
    t = mod(t,35.0);

    vec3 u = UP;
    float f = 1.0;
    if(t<PI4)
    {
        e = vec3(30.0*cos(t*.125),2.0,30.0*sin(t*.125));
        d = normalize(vec3(0.0)-e);
        s = shutterfade(0.0, PI4, t, .5);
    }
    else if(t<20.0)
    {
        e = mix(PLANET_ROT*-10.0,PLANET_ROT*10.0,smoothstep(PI4,20.0,t));
        e.y += 1.0;
        d = PLANET_ROT;
        s = shutterfade(PI4, 20.0, t, .5);
    }
    else if(t<25.0)
    {
        e = mix(vec3(-10.0,1.0,3.0),vec3(10.0,1.0,3.0),smoothstep(20.0,25.0,t));
        d = vec3(0.948683, 0.316228, 0.0);
        u = vec3(-d.y,d.x,0.0);
        s = shutterfade(20.0, 25.0, t, .5);
    }
    else if(t<30.0)
    {
        e = mix(vec3(-30.0,10.0,-10.0),vec3(10.0,10.0,10.0),smoothstep(25.0,30.0,t));
        d = vec3(0.666667, -0.333333, 0.666667);
        u = vec3(-d.y,d.x,0.0);
        s = shutterfade(25.0, 30.0, t, .5);
    }
    else
    {
        e = mix(vec3(1.0,1.0,2.0),vec3(1.0,5.0,1.5),smoothstep(30.0,35.0,t));
        d = UP;
        u = PLANET_ROT;
        f = .5;
        s = shutterfade(30.0, 35.0, t, .5);
    }
    camera(uv, e, d, u, f, p, d);
}

/*
	Takes a vec2 containing a distance and a primitive's ID and returns
	the ID and distance of the nearer primitive, effectively performing
	a solid modeling union.
*/
vec2 u( in vec2 a, in vec2 b )
{
    return mix(a,b,step(b.s,a.s));
}

/*
	The random function from the first Stack Overflow article that is
	returned after googling "GLSL noise".
	Can we have a conversation about how GLSL defines a noise function
	but no vendors actually implement it?
*/
float rand( in vec2 co )
{
    return fract(sin(dot(co,RAND_SEED)) * 43758.5453);
}

/*
	A 2D texture-sampled noise function. Used when the surface to be
	distorted exists entirely in one plane.
*/
float n(vec2 p, sampler2D tex)
{
    return texture(tex,p).r;
}

/*
	IQ's seminal noise function. I learned that the weird ring artifacts
	were due to vflipping it's texture buddy.
*/
float nIQ( in vec3 x, in sampler2D t )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);

    vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
    vec2 rg = texture( t, (uv+0.5)/256.0, -100.0 ).yx;
    return mix( rg.x, rg.y, f.z );
}

/*
	A 2D FBM with time dependence.
*/
float fbm2Dt( in vec2 p, in float t, in sampler2D tex )
{
    float r = n(p+t*.1,tex);
    p*=ROT45;p*=2.0;p+=t*-.24;
    r += n(p,tex)*.500;
    p*=ROT45;p*=2.0;p+=t*-.12;
    #ifdef HQ_RENDERING
    r += n(p,tex)*.250;
    #endif
    return r*1.0075;
}

/*
	A 3 octave FBM that uses IQ's noise.
*/
float fbm3D( in vec3 p, in sampler2D t )
{
    float r = nIQ(p, t)*.5;
    r += nIQ(p*2.0, t)*.25;
    r += nIQ(p*4.1, t)*.125;
    return r * 1.11872;
}


/*
	Returns the distance to the ocean plane.
*/
float ocean( in vec3 p )
{
    // Translate to the Y coordinate of the ocean.
    p.y -= OCEAN_HEIGHT;
    // Do some optimization where if we're beyond a certain
    // distance from the plane, we don't worry about calculating the bump height.
    // The *4.0 is there because the bump map would fall to 0 near the horizon.
    if(p.y>OCEAN_AMPLITUDE*4.0)return p.y;
    // If we're within that threshold, we modulate the distance by the bump map.
    else return p.y+fbm2Dt(p.xz*.025, iTime*OCEAN_SPEED, OCEAN_NOISE_TEX)*OCEAN_AMPLITUDE;
}

/*
	Returns the distance to the sand. This is really just sphere that looks cool
	and is drowning.
*/
float sand( in vec3 p )
{
    // Translate the sphere down into the water.
    p.y += 50.0;
    // An inlining of the classic sphere distance formula.
    float sphere = length(p)-51.0;
    // Same distance check as with the ocean.
    if(sphere-SAND_BUMP_HEIGHT> 0.0) return sphere;
    // Same method of height modulation. Since we're only considering the
    // scalp of the sphere, we don't need worry about the fact that we're
    // using a 2D fbm.
    else
    {
        return sphere + fbm2Dt(p.xz*.025,0.0,SAND_NOISE_TEX)
            * SAND_BUMP_HEIGHT;	// The general height of the sand features.
    }
}

/*
	The distance function of the rock.
*/
float rock( in vec3 p )
{
    // Inlined torus distance function.
    vec2 q = vec2(length(p.xy)-ROCK_DIM.x,p.z);
    float d = length(q)-ROCK_DIM.y;

    // The sum of the heights of the FBM and noise calls.
    if(d > 1.5125) return d;
    else
    {
        float t = fbm3D(p, RAND_NOISE_TEX);
        #ifdef HQ_RENDERING
        t += nIQ(p*1.0,RAND_NOISE_TEX)*.5;
        #endif
        t += nIQ(p*32.0,RAND_NOISE_TEX)*.0125;
        return d + t;
    }
}


/*
	The scene's distance function.
*/
float dist( in vec3 p )
{
    // I think nesting this is a bit more better than dicing it out
    // into two calls. Could someone confirm or deny this?
    return min(sand(p),min(ocean(p),rock(p)));
}

/*
	The distance function of the stuff that's visible in
	reflections. This doesn't include some stuff to save
	time.
*/
float distR( in vec3 p )
{
    // By not having the ocean here we're cutting out hundreds
    // of texture samples per ray.
    return min(sand(p),rock(p));
}

/*
	Returns the id and distance of the nearest object.
*/
vec2 distID( in vec3 p )
{
    vec2 d = u(vec2(sand(p),ID_SAND), vec2(ocean(p), ID_OCEAN));
    return u(d, vec2(rock(p), ID_ROCK));
}

/*
	Another reflection shortcut variation of another function.
*/
vec2 distIDR( in vec3 p )
{
    return u(vec2(sand(p),ID_SAND),vec2(rock(p), ID_ROCK));
}

/*
	Returns the direction of the sun based on the current time.
*/
vec3 sunDir()
{
    return rodRot(SUN_DIR, PLANET_ROT, iTime*TIME_WARP);
}

/*
	Returns the color of the sun. Just does some janky fading
	between white at zenith and yellow at asimuth.
*/
vec3 sunColor( in vec3 sunDir )
{
    return mix(SUN_COLOR_B, SUN_COLOR_A, clamp(dot(sunDir,UP),0.0,1.0));
}

/*
	Okay this is not my atmospheric scattering solution, and for the
	life of me I can't find the shader that I'm borrowing it from.
	It's an implementation of the method discussed in this paper:
	http://www.cs.utah.edu/~shirley/papers/sunsky/sunsky.pdf

	The nice thing is that it's not the usual Scratchapixel solution.
*/
float saturatedDot( in vec3 a, in vec3 b )
{
    return max( dot( a, b ), 0.0 );
}
vec3 YxyToXYZ( in vec3 Yxy )
{
    float X = Yxy.g * ( Yxy.r / Yxy.b );
    float Z = ( 1.0 - Yxy.g - Yxy.b ) * ( Yxy.r / Yxy.b );

    return vec3(X,Yxy.r,Z);
}
vec3 XYZToRGB( in vec3 XYZ )
{
    // CIE/E
    return XYZ * mat3
    (
        2.3706743, -0.9000405, -0.4706338,
        -0.5138850,  1.4253036,  0.0885814,
        0.0052982, -0.0146949,  1.0093968
    );
}
vec3 YxyToRGB( in vec3 Yxy )
{
    vec3 XYZ = YxyToXYZ( Yxy );
    vec3 RGB = XYZToRGB( XYZ );
    return RGB;
}
void calculatePerezDistribution( in float t, out vec3 A, out vec3 B, out vec3 C, out vec3 D, out vec3 E )
{
    A = vec3(  0.1787 * t - 1.4630, -0.0193 * t - 0.2592, -0.0167 * t - 0.2608 );
    B = vec3( -0.3554 * t + 0.4275, -0.0665 * t + 0.0008, -0.0950 * t + 0.0092 );
    C = vec3( -0.0227 * t + 5.3251, -0.0004 * t + 0.2125, -0.0079 * t + 0.2102 );
    D = vec3(  0.1206 * t - 2.5771, -0.0641 * t - 0.8989, -0.0441 * t - 1.6537 );
    E = vec3( -0.0670 * t + 0.3703, -0.0033 * t + 0.0452, -0.0109 * t + 0.0529 );
}
vec3 calculateZenithLuminanceYxy( in float t, in float thetaS )
{
    float chi  	 	= ( 4.0 / 9.0 - t / 120.0 ) * ( PI - 2.0 * thetaS );
    float Yz   	 	= ( 4.0453 * t - 4.9710 ) * tan( chi ) - 0.2155 * t + 2.4192;

    float theta2 	= thetaS * thetaS;
    float theta3 	= theta2 * thetaS;
    float T 	 	= t;
    float T2 	 	= t * t;

    float xz =
    ( 0.00165 * theta3 - 0.00375 * theta2 + 0.00209 * thetaS + 0.0)     * T2 +
    (-0.02903 * theta3 + 0.06377 * theta2 - 0.03202 * thetaS + 0.00394) * T +
    ( 0.11693 * theta3 - 0.21196 * theta2 + 0.06052 * thetaS + 0.25886);

    float yz =
    ( 0.00275 * theta3 - 0.00610 * theta2 + 0.00317 * thetaS + 0.0)     * T2 +
    (-0.04214 * theta3 + 0.08970 * theta2 - 0.04153 * thetaS + 0.00516) * T +
    ( 0.15346 * theta3 - 0.26756 * theta2 + 0.06670 * thetaS + 0.26688);

    return vec3( Yz, xz, yz );
}
vec3 calculatePerezLuminanceYxy( in float theta, in float gamma, in vec3 A, in vec3 B, in vec3 C, in vec3 D, in vec3 E )
{
    return ( 1.0 + A * exp( B / cos( theta ) ) ) * ( 1.0 + C * exp( D * gamma ) + E * cos( gamma ) * cos( gamma ) );
}
vec3 calculateSkyLuminanceRGB( in vec3 s, in vec3 e, in float t )
{
    vec3 A, B, C, D, E;
    calculatePerezDistribution( t, A, B, C, D, E );
    float thetaS = acos( saturatedDot( s, vec3(0,1,0) ) );
    float thetaE = acos( saturatedDot( e, vec3(0,1,0) ) );
    float gammaE = acos( saturatedDot( s, e )		   );
    vec3 Yz = calculateZenithLuminanceYxy( t, thetaS );
    vec3 fThetaGamma = calculatePerezLuminanceYxy( thetaE, gammaE, A, B, C, D, E );
    vec3 fZeroThetaS = calculatePerezLuminanceYxy( 0.0,    thetaS, A, B, C, D, E );
    vec3 Yp = Yz * ( fThetaGamma / fZeroThetaS );
    return YxyToRGB( Yp );
}

/*
	Takes a ray direction (presumably out into the sky)
	and returns how much it hits a star.
*/
float stars( in vec3 d)
{
    d = rodRot( d, PLANET_ROT, -iTime*TIME_WARP);
    return pow(nIQ(d*200.0,RAND_NOISE_TEX),80.0); // These numbers are so magic they
    // could carry the Lakers.
}

/*
	Combines the sky radiance from the magic above with a specular
	highl^H^H^H^H^H^Hsun.
*/
vec3 sky( in vec3 d, in vec3 ld )
{
    // Get the sky color.
    vec3 sky = calculateSkyLuminanceRGB(ld, d, 3.0);

    // How night time is it? This variable will tell you.
    float night = smoothstep(-0.0, -0.5, clamp(dot(ld, UP),-0.5, -0.0));
    // Set a general brightness level so we don't just have a white screen,
    // and artificially darken stuff at night so it looks good.
    sky *= .040-.035*night;

    // Create a spot for the sun. This version gives us some nice edges
    // without having a pow(x,VERY_LARGE_NUMBER) call.
    float sunspot = smoothstep(.99935, .99965, max(dot(d,ld),0.0));
    sunspot += smoothstep(.98000, 1.0, max(dot(d,ld),0.0))*.05; // Corona.

    // Mix the sky with the sun.
    sky = sky*(1.0+sunspot);

    // Also add in the stars.
    return sky + stars(d)*.5;
}

/*
	Returns the surface normal of the distance field at the given
	point p.
*/
vec3 norm( in vec3 p )
{
    return normalize(vec3(dist(vec3(p.x+EPSILON,p.y,p.z)),
        dist(vec3(p.x,p.y+EPSILON,p.z)),
        dist(vec3(p.x,p.y,p.z+EPSILON)))-dist(p));
}

/*
	Returns the surface normal of the stuff in reflections.
*/
vec3 normR( in vec3 p )
{
    return normalize(vec3(distR(vec3(p.x+EPSILON,p.y,p.z)),
        distR(vec3(p.x,p.y+EPSILON,p.z)),
        distR(vec3(p.x,p.y,p.z+EPSILON)))-distR(p));
}

/*
	The ray-marching function. Marches a point p along a direction d
	until it reaches a point within a minimum distance of the distance
	field.
*/
void march( inout vec3 p, in vec3 d, in vec3 e )
{
    float r = dist(p+d*EPSILON);
    for(int i = 0; i < V_STEPS; i++)
    {
        if(r < EPSILON || length(p-e) > MAX_DEPTH)
            return;
        p += d*r*.66; // The higher levels of noise in the rock may be skipped
        // if the steps are too large.
        r = dist(p);
    }
    return;
}

/*
	Yet another secondary function for the case of reflections. This one
	does ray marching.
*/
void marchR( inout vec3 p, in vec3 d, in vec3 e )
{
    float r = distR(p+d*EPSILON);
    for(int i = 0; i < R_STEPS; i++)
    {
        if(r < EPSILON || length(p-e) > MAX_DEPTH)
            return;
        p += d*r; // Eh the reflections are barely seen. We don't need to
        // attenuate distance marched because of high frequency distance stuff.
        r = distR(p);
    }
    return;
}

/*
	March-from-surface-to-light shadowing, with IQ's glancing penumbras.
*/
float shadow( in vec3 start, in vec3 n, in vec3 ldir, in float p )
{
    // Do some quick "is the sun even shining on here" tests.
    // We wait until the sun is just below the horizon before considering
    // it gone.
    if( dot(n,ldir) <= 0.0 || dot(ldir,UP) <= -.25) return 0.0;

    float t = EPSILON*40.0;
    float res = 1.0;
    for ( int i = 0; i < S_STEPS; ++i )
    {
        float d = distR( start + ldir * t );
        if ( d < EPSILON*.1 )
            return 0.0;

        res = min( res, p * d / t );
        t += d;

        if ( t > MAX_DEPTH )
            break;
    }
    return res;
}

/*
	IQ's really compact implementation of Oren Nayar reflectance.
*/
float orenNayar( in vec3 n, in vec3 v, in vec3 ldir )
{
    float r2 = pow(MAT_REFLECTANCE, 2.0);
    float a = 1.0 - 0.5*(r2/(r2+0.57));
    float b = 0.45*(r2/(r2+0.09));

    float nl = dot(n, ldir);
    float nv = dot(n, v);

    float ga = dot(v-n*nv,n-n*nl);

    return max(0.0,nl) * (a + b*max(0.0,ga) * sqrt((1.0-nv*nv)*(1.0-nl*nl)) / max(nl, nv));
}

/*
	Calculates the ambient occlusion factor at a given point in space.
	Uses IQ's marched normal distance comparison technique.
*/
float occlusion( in vec3 pos, in vec3 norm)
{
    // I even unrolled it for you. Aren't I nice?
    float result = pow(2.0,-1.0)*(0.25-dist(pos+0.25*norm));
    result += pow(2.0,-2.0)*(0.50-dist(pos+0.50*norm));
    result += pow(2.0,-3.0)*(0.75-dist(pos+0.75*norm));
    return 1.0-result*OCC_FACTOR;
}

/*
	Performs the full suite of lighting for Oren Nayar-appropriate
	surfaces. Calculates direct (sun), sky, and ambient radiance
	contributions.
*/
vec3 light( in vec3 p, in vec3 d, in vec3 e, in vec3 n, in vec3 ld )
{
    // Get an ambient occlusion value.
#ifdef CALC_OCCLUSION
    float amb = occlusion(p,n)*max(dot(ld,UP),0.25);
    #else
    float amb = 1.0;
    #endif

    // Get light colors and radiance for the three lights.
    // (Or just specral radiance if that's your thing.)
    vec3 skc = .6*orenNayar(n,-d,UP)*sky(n,ld);
    vec3 sun = orenNayar(n,-d,ld)*sunColor(ld);
    vec3 gnd = .2*GROUND_COLOR*max(orenNayar(n,-d,-UP),0.25);

    // Modulate those by ambient occlussion and shadowing.
    skc *= amb;
    gnd *= amb;
    sun *= shadow(p,n,ld,PENUMBRA_FACTOR);;

    // Return the sum.
    return skc+gnd+sun;
}

/*
	Returns the texture of the sand, and does some basic distance
	filtering.
*/
vec3 texSand( in vec3 p, in float l )
{
    // Essentially what we're doing is just calculating low contrast
    // random values whose contrast diminishes with distance.
    return SAND_COLOR*mix(.75+rand(p.xz)*.25, .85, clamp(l/MAX_DEPTH,0.0,1.0));
}

/*
	Shades dry sand. Just takes in an illumination value and texture
	and multiplies the two together. Simple.
*/
vec3 shadeDrySand( in vec3 i, in vec3 t )
{
    return i*t;
}

/*
	Non reflective wet sand is a bit harder, but still simple.
	Essentially we're doing per-pixel Phong shading, but the diffuse
	term is .5 of the dry sand result. Oh, and the ambient term is
	the sky above.
*/
vec3 shadeWetSand( in vec3 p, in vec3 d, in vec3 e, in vec3 n, in vec3 i, in vec3 ld, in vec3 tex )
{
    // Specular.
    vec3 r = reflect(d,n);
    vec3 spec = sunColor(ld) * max( pow(dot(ld,r),50.0), 0.0);
    // Diffuse.
    vec3 diff = shadeDrySand(i,tex)*.5;
    // Ambient.
    vec3 ambi = sky(n,ld);
    return spec+diff+ambi;
}

/*
	Constructs the appearance of non-reflection sand (sand that is
	already a reflection.) Since you barely see this sand, we don't
	worry about if it's wet or not. (To save cycles.)
*/
vec3 shadeSand( in vec3 p, in vec3 i, in float l )
{
    // Get the texture of the sand.
    vec3 tex = texSand(p,l);
    // Pass it into the the shading function.
    return shadeDrySand(i,tex);
}

/*
	Prepares the texture of the rock. It's pretty much just
	a jumbled mix of two colors.
*/
vec3 texRock( in vec3 p )
{
    p.y *= 8.0;
    float chroma = nIQ(p*8.0, RAND_NOISE_TEX);
    // Mix the two rock colors based on a noise value, then somewhat
    // randomize the lumosity to make it look rough.
    return mix(ROCK_COLOR_A, ROCK_COLOR_B,chroma)*mix(.8, 1.2, rand(p.xy));
}

/*
	Shades the rock. With the illumination color and value pre-computed
	this is a simple task.
*/
vec3 shadeRock( in vec3 p, in vec3 i )
{
    return texRock(p)*i;
}

/*
	Shades the scene as seen through a reflection.
	Note how reflections only feature a subset of the
	scene's items.
*/
vec3 shadeR( in vec3 p, in vec3 d, in vec3 ld, in vec3 e )
{
    // Get the distance to the point.
    float l = length(p-e);

    // If the ray ran out of length, we return the sky.
    if(l > MAX_DEPTH) return sky(d,ld);

    // Get the ID of the current object.
    float id = distID(p).t;
    // Get the surface normal of the reflected scene.
    vec3 n = normR(p);
    // Get a general light color to use...wherever.
    vec3 i = light(p,d,e,n,ld);

    // Do some piecewise shading.
#ifdef HQ_RENDERING
    if(id == ID_ROCK) return shadeRock(p,i);
    else 			  return shadeSand(p,i,l);
    #else
    return shadeSand(p,i,l);
    #endif
}

/*
	Constructs the appearance of reflective sand.
*/
vec3 shadeWetSandR( in vec3 p, in vec3 d, in vec3 e, in vec3 n, in vec3 i, in vec3 ld, in vec3 tex )
{
    // Get the reflected ray direction.
    vec3 r = reflect(d,n);

    // March the (now reflected) ray in that direction.
#ifdef HQ_RENDERING
    marchR(p,r,e);
    // Get the appearance of the point the reflect ray found.
    vec3 reflection = shadeR(p,r,ld,e);
    #else
    vec3 reflection = sky(r,ld);
    #endif
    // Create a standard Phong specular term to spice things up in
    // the shiny department.
    vec3 specular = pow(max(dot(r,ld),0.0),350.0)*sunColor(ld);
    // Get the reflectance of the surface given the angle of incidence.
    float reflectance = max(length(cross(n,d)),0.0);
    // Get the diffuse term. (discussed in shadeWetSand().)
    vec3 diffuse = tex*i*.5;
    // Return a blend of the reflection and the sand and the specular highlight.
    return mix(diffuse, reflection, pow(reflectance,128.0))+specular;
}

/*
	Constructs the appearance of first-bounce sand, whether it's wet and reflective
	or not.
*/
vec3 shadeSandR( in vec3 p, in vec3 d, in vec3 e, in vec3 n, in vec3 i, in vec3 ld, in float l )
{
    // Compute the texture of the sand.
    vec3 tex = texSand(p,l);

    // Figure out how wet the sand is. This is just some quasi-good value
    // that is vaguely related to the height of the water, but higher.
    float wetness = clamp(smoothstep(OCEAN_HEIGHT, 0.075, p.y),
    0.0,1.0);

    // Get the appearance of reflective wet sand.
    vec3 wet = shadeWetSandR(p,d,e,n,i,ld,tex);
    // Get the appearance of the dry sand.
    vec3 dry = shadeDrySand(i,tex);

    // Return a mix of wet and dry sand based on the wetness of the sand.
    return mix(wet,dry,wetness);
}

/*
	Shades the ocean with its reflections.
*/
vec3 shadeOceanR( in vec3 p, in vec3 d, in vec3 e, in vec3 n, in vec3 ld, in vec3 sky )
{
    // Get the pirate vector.
    vec3 r = reflect(d,n);

    // March P away from where it started out.
    marchR(p,r,e);

    // Get the image of the reflection.
    vec3 reflection = shadeR(p,r,ld,e);
    // Make a swanky specular highlight.
    vec3 specular = pow(max(dot(r,ld),0.0),500.0)*sunColor(ld);

    // Get a non-reflection color for the ocean.
    vec3 ocean = vec3(.0,.08, .1)*max(0.005, dot(d,UP));
    // Get how much the reflection should be visible.
    float reflectance = pow(max(length(cross(n,d)),0.0),2.0); //Sin^2(incidence)
    // Do some mixin'.
    return mix(ocean,reflection,reflectance*.5)+specular;
}

/*
	The first bounce shading function. Takes that first ray, beats it up,
	and sends to next Tuesday.
*/
vec3 shade( in vec3 p, in vec3 d, in vec3 e )
{
    // Get the distance for that gentle blend at the horizon.
    float l = length(p-e);
    // Get the sun direction and sky color once, and pass it around.
    vec3 ld = sunDir();
    vec3 s = sky(d,ld);

    // Take care of the appearance of culled stuff--they're sky now.
    if(l > MAX_DEPTH) return s;

    // Get the ID of the object we're shading, and make a place
    // to store the final result.
    float id = distID(p).t;
    vec3 result;

    // Get the normal and create a general Oren Nayar term to be
    // the shade functions' village bicycle, right next to the
    // light direction, which is the village Razor scooter.
    vec3 n = norm(p);
    vec3 i = light(p,d,e,n,ld);

    // Shade the surface, based on the object ID.
    if(id == ID_OCEAN){
        float distSand = sand(p);
        float distOcean = ocean(p);
        //vec3 sand = shadeSandR(p,d,e,n,i,ld,l);
        //vec3 ocean = shadeOceanR(p,d,e,n,ld,s);
        result = mix(sand,ocean,clamp(distSand-distOcean,0.0,2.0)*.5);
    }
    else if(id == ID_SAND)	result = shadeSandR(p,d,e,n,i,ld,l);
    else result = shadeRock(p,i);

    // Mix this shaded color with the color of the sky at far distances,
    // so we don't have that ugly plane() @ MAX_DEPTH edge.
    return s;
}

/*
	Performs some quick post-processing. Does gamma correction
	and adds a soft vignette just to make whatever you're doing
	look pretty.
*/
vec3 postProcess( in vec2 uv, in vec3 c )
{
    float vig = 1.0-dot(uv,uv)*.3;
    return pow(c,vec3(1.0/2.2))*vig;
}

/*
	Shadertoy's entry point.
*/
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Standard pixel-coordinate normalization.
    vec2 uv = fragCoord / iResolution.xy - 0.5;
    uv.x *= iResolution.x/iResolution.y; //fix aspect ratio

    // Position, direction, and eye.
    // Notice that it spells out the Processing file extension?
    vec3 p,d,e;

    // The shutter. This is populated by animateCam() as a frame
    // brightness coefficient to fade between scenes.
    float s;

    // Set up the camera.
    animateCam(uv,iTime,p,d,e,s);

    // Do the actual ray marching.
    march(p,d,e);

    // Store the final pixel color.
    //fragColor = postProcess(uv,shade(p,d,e)).rgbb*s;
    // Eventually the alpha of fragColor may be used.
    fragColor = vec4(postProcess(uv,shade(p,d,e)),1.0);
}