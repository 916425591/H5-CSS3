const float EPS = 0.001;

const float pi = 3.14159265359;

//2d Fast Clouds const
//SETTINGS//
const float timeScale = 10.0;
const float cloudScale = 0.5;
const float skyCover = 0.6; //overwritten by mouse x drag
const float softness = 0.2;
const float brightness = 1.0;
const int noiseOctaves = 8;
const float curlStrain = 3.0;
//SETTINGS//

mat3 ry (float radian){
    return mat3(cos(radian), 0.0,-sin(radian),
        0.0, 1.0, 0.0,
        sin(radian), 0.0, cos(radian)  );
}

mat3 rx (float radian){
    return mat3(1.0, 0.0, 0.0,
        0.0, cos(radian), sin(radian),
        0.0,-sin(radian), cos(radian) );
}
//Random function
float rand(vec2 n) {
    return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}
//Noise function
float noise(vec2 n) {
    const vec2 d = vec2(0.0, 1.0);
    vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
    return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}
//Fbm function
float fbm(vec2 n) {
    float total = 0.0, amplitude = 1.0;
    for (int i = 0; i < 8; i++) {
        total += noise(n) * amplitude;
        n += n;
        amplitude *= 0.5;
    }
    return total;
}


float distanceFunction(vec3 p){
    vec4 n = vec4(0.0,1.0,0.0,1.0);
    float disp;
    disp = .3*sin(1.4*p.x+cos(p.z*1.3))-(1.0-abs(sin(p.x+cos(p.z+fbm(p.zx)))))*.4;
    float plane = dot(p,n.xyz) + n.w;

    return (plane+disp);
}

vec3 getNormal(vec3 p){
    const float d = EPS;
    return
    normalize
    (
        vec3
        (
            distanceFunction(p+vec3(d,0.0,0.0))-distanceFunction(p+vec3(-d,0.0,0.0)),
            distanceFunction(p+vec3(0.0,d,0.0))-distanceFunction(p+vec3(0.0,-d,0.0)),
            distanceFunction(p+vec3(0.0,0.0,d))-distanceFunction(p+vec3(0.0,0.0,-d))
        )
    );
}

float saturate(float num)
{
    return clamp(num,0.0,1.0);
}

float noised(vec2 uv)
{
    return texture(iChannel2,uv).r;
}

vec2 rotate(vec2 uv)
{
    uv = uv + noised(uv*0.2)*0.005;
    float rot = curlStrain;
    float sinRot=sin(rot);
    float cosRot=cos(rot);
    mat2 rotMat = mat2(cosRot,-sinRot,sinRot,cosRot);
    return uv * rotMat;
}

float fbm2 (vec2 uv)
{
    float rot = 1.57;
    float sinRot=sin(rot);
    float cosRot=cos(rot);
    float f = 0.0;
    float total = 0.0;
    float mul = 0.5;
    mat2 rotMat = mat2(cosRot,-sinRot,sinRot,cosRot);

    for(int i = 0;i < noiseOctaves;i++)
    {
        f += noised(uv+iTime*0.00015*timeScale*(1.0-mul))*mul;
        total += mul;
        uv *= 3.0;
        uv=rotate(uv);
        mul *= 0.5;
    }
    return f/total;
}
void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 pos = (fragCoord.xy*2.0 -iResolution.xy) / iResolution.y;
    vec3 bg = vec3(0.0);//mix(vec3(0.24,0.24,.9),vec3(1.5),.3);

    float time = iTime*.3;
    vec3 camPos = vec3(0.0, -.25, 11.0)*ry(time*.15);
    camPos.y=-distanceFunction(camPos);
    //Camera set-up
    vec3 camDir = vec3(.3, 0.05, -1.0);
    vec3 camUp = vec3(0.0, 1.0, 0.0);
    vec3 camSide = cross(camDir, camUp);
    float focus = 1.8;
    vec3 col = vec3(0.0);
    vec3 rayDir = normalize(camSide*pos.x + camUp*pos.y + camDir*focus)*ry(iMouse.x/51.)*ry(time*.15-1.25);

    float t = 0.0, d;
    vec3 posOnRay = camPos;
    //Raymarching
    for(int i=0; i<48; ++i){
        d = distanceFunction(posOnRay);
        t += d;
        posOnRay = camPos + t*rayDir;
        if(d < EPS) break;
    }

    vec4 tex = texture(iChannel0,posOnRay.xz*.25);
    vec3 l = normalize(vec3(0.0,10.,-20.)*ry(time));
    vec3 normal = getNormal(posOnRay);

    //sun-sunIntensity from 'Kepler 256o'
    //https://www.shadertoy.com/view/XsjGRd - otaviogood
    vec3 localRay = normalize(rayDir);
    float sunIntensity = 1.0 - (dot(localRay, l) * 0.5 + 0.5);
    //sunIntensity = (float)Math.Pow(sunIntensity, 14.0);
    sunIntensity = 0.2 / sunIntensity;
    sunIntensity = min(sunIntensity, 40000.0);
    sunIntensity = max(0.0, sunIntensity - 3.0);
    //////////////////////////////////////////////////
    vec2 screenUv = fragCoord.xy/iResolution.xy;
    vec2 uv2 = fragCoord.xy/(40000.0*cloudScale);

    float mouseXAffect = (iMouse.x/iResolution.x);
    float weatherVariation = clamp(sin(time)*.5,-0.5,-0.05);
    float cover = 0.5-weatherVariation;

    float bright = brightness*(1.8-cover);
    //cloud coverage from '2d Fast Clouds'
    //https://www.shadertoy.com/view/XsjSRt - Sinuousity

    //perspective correction from Dave Hoskins 'Mountains' GetColor function
    ////////////////
    vec3 rd = rayDir; rd.y = max(rd.y, 0.0001);
    float v = (7.5-camPos.y)/rd.y;
    rd.xz *= v;
    rd.xz += camPos.xz;
    rd.xz *= .010;
    ////////////////

    //float color1 = fbm2((rayDir.yz*.05)-0.5+iTime*0.00004*timeScale);  //xz
    //float color2 = fbm2((rayDir.zy*.05)-10.5+iTime*0.00002*timeScale); //yz
    float color1 = fbm2((rd.xz*.05)-0.5+iTime*0.00004*timeScale);  //xz
    float color2 = fbm2((rd.zx*.05)-10.5+iTime*0.00002*timeScale); //yz

    float clouds1 = smoothstep(1.0-cover,min((1.0-cover)+softness*2.0,1.0),color1);
    float clouds2 = smoothstep(1.0-cover,min((1.0-cover)+softness,1.0),color2);

    float cloudsFormComb = saturate(clouds1+clouds2);

    vec4 skyCol = vec4(0.6,0.8,1.0,1.0);
    float cloudCol = saturate(saturate(1.0-pow(color1,1.0)*0.2)*bright);
    vec4 clouds1Color = vec4(cloudCol,cloudCol,cloudCol,1.0);
    vec4 clouds2Color = mix(clouds1Color,skyCol,0.25);
    vec4 cloudColComb = mix(clouds1Color,clouds2Color,saturate(clouds2-clouds1));
    vec4 clouds = vec4(0.0);
    clouds = mix(skyCol,cloudColComb,cloudsFormComb);
    vec3 sunCol = vec3(258.0, 208.0, 100.0) / 15.0;
    //bg;  //dot(l,rayDir)*.5+.2;
    bg=clouds.rgb;
    if(abs(d) < 0.5){
        //Diffuse
        float diff = clamp(dot(normal,(l)),0.0,1.0);
        vec3 brdf = 1.5*vec3(.10, .11, .11);
        float fre = .2*pow(clamp(1. + dot(normal, rayDir), 0., 1.), 2.);
        brdf += 1.30*diff*vec3(1., .9, .75);
        //Blinn-Phong half vector
        vec3 h = normalize(-rayDir + l);
        //Specular
        float spe = pow(clamp(dot(h, normal), 0.0, 1.0), 15.0*4.);
        //Textured specular
        vec4 spec = texture(iChannel1,posOnRay.xz)*spe;
        //Adding diffuse and specular
        col=diff*tex.rgb + spec.rgb+fre*brdf;
        //Fog
        fragColor = vec4(mix(col,bg,smoothstep(.01,.2,t*.015)),1.0);
    } else {
        //Sky
        fragColor = vec4(bg+sunCol*(sunIntensity*.0015),1.0);
    }
}