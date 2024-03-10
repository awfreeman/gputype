#version 450


#define PI 3.14159
layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

const int scans = 25;


layout(set = 0, binding = 0) buffer curve_buf {
    mat3x2[] curves;
};
layout(set = 0, binding = 1) buffer len_buf {
    uint len;
};

vec2 bezqpt(float t, mat3x2 p) {
    return (pow((1.-t), 2.)*p[0]) + (((2.*t)*(1.-t))*p[1]) + (pow(t, 2.)*p[2]);
}


vec2 bezq_roots(mat3x2 p) {
    float c = p[0].y;
    float b = 2.*(p[1].y- p[0].y);
    float a = p[2].y + (-2.*p[1].y) + p[0].y;
    float under_root = (b*b) - (4.*a*c);
    if (under_root < 0.) {
        return vec2(1./0.);
    }
    float root = pow(under_root, 0.5);
    vec2 res = vec2((2.*c)/((-b)+root), (2.*c)/((-b)-root));
    return res;
}

vec2 bezq_deriv(float t, mat3x2 p) {
    return (2.*(1.-t)*(p[1]-p[0])) + (2.*t*(p[2]-p[1]));
}

float sqd(float t, mat3x2 p, vec2 pt) {
    return pow(distance(bezqpt(t, p), pt), 2.);
}

float lmin(float lower, float upper, mat3x2 p, vec2 pt) {
    float e = 1./pow(10., 4.);
    float m = lower;
    float n = upper;
    float k;
    while ((n-m) > e) {
        k = (n+m)/2.;
        sqd((k-e), p, pt) < sqd((k+e), p, pt) ? n=k : m=k;
    }
    return k;

}

void outline() {
    vec2 tc = tex_coords;
    float color = 0.;
    for (uint i = 0; i < len; i++) {
        mat3x2 p = curves[i];
        for (int x = 0; x < 3; x++) {
            if (distance(tc, p[x]) < 0.007) {
                f_color.y = 1.;
            }
        }
        int min_num = 0;
        float min_dst = 1./0.;
        for(int i = 0; i <= scans; i++) {
            float fr = float(i)/float(scans);
            float dst = sqd(fr, p, tc);
            if (dst < min_dst) {
                min_dst = dst;
                min_num = i;
            }
        }
        float min_t = lmin(clamp( (float(min_num - 1)/float(scans)), 0., 1.), clamp((float(min_num + 1)/float(scans)), 0., 1.), p, tc);
        color += smoothstep(0.005, 0.0, distance(bezqpt(min_t, p), tc));
    }
    f_color.z +=color;
}
void solid() {
    vec2 tc = tex_coords;
    float color = 0.;
    int ctr = 0;
    for (uint i = 0; i < len; i++) {
        mat3x2 p = curves[i];
        p[0] -= tc;
        p[1] -= tc;
        p[2] -= tc;
        vec2 roots = bezq_roots(p);
        vec2 rx = bezqpt(roots.x, p);
        vec2 ry = bezqpt(roots.y, p);
        if (roots.x < 1. && roots.x > 0. && rx.x > 0. ) {
            vec2 derivative = bezq_deriv(roots.x, p);
            if (derivative.y > 0.) {
                ctr += 1;
            } else {
                ctr -= 1;
            }
        }
        if (roots.y < 1. && roots.y > 0. && ry.x > 0. ) {
            vec2 derivative = bezq_deriv(roots.y, p);
            if (derivative.y > 0.) {
                ctr += 1;
            } else {
                ctr -= 1;
            }
        }
    }
    if (ctr != 0) {
        f_color.xyz = vec3(0.5);
    }
}
void main() {
    f_color = vec4(0.);
    vec2 tc = tex_coords;
    if (tc.x > 1. || tc.x < 0.) {
        return;
    }
    if (tc.y > 1. || tc.y < 0.) {
        return;
    }
    solid();
    /*outline();
    if (distance(tc, vec2(0.25, 0.92)) < 0.01) {
        f_color.x = 1;
    }*/
}