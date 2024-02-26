#include "stdio.h"
#include "math.h"


typedef struct vec2{
    float x;
    float y;
} vec2;


vec2 bezq_roots(vec2 p[3]) {
    vec2 res;
    float c = p[0].y;
    float b = 2.*p[1].y;
    float a = p[2].y + (-2.*p[1].y) - p[0].y;
    float under_root = (b*b) - (4.*a*c);
    if (under_root < 0.) {
        res.x = INFINITY;
        res.y = INFINITY;
        puts("NEGATIVE ROOT");
        return res;
    }
    printf("%f\n", under_root);
    float root = sqrt(under_root);
    printf("%f\n", root);
    res.x = ((-b) + root) / (2.*a);
    res.y = ((-b) - root) / (2.*a);
    return res;
}

vec2 bezq_pt(vec2 curve[3], float t) {
    float a = 1. - (t*t);
    float b = (2.*t) * (1. - t);
    float c = t*t;
    vec2 res;
    res.x = (curve[0].x*a) + (curve[1].x * b) + (curve[2].x * c);
    res.y = (curve[0].y*a) + (curve[1].y * b) + (curve[2].y * c);
    return res;
}

vec2 nvec2(float x, float y) {
    vec2 res;
    res.x = x;
    res.y = y;
    return res;
}
vec2 sub(vec2 rhs, vec2 lhs) {
    rhs.x -= lhs.x;
    rhs.x -= lhs.x;
    return rhs;
}
//[0.046511628, 0.73975044, 0.09108527, 1.0, 0.45639536, 1.0]
void main() {
    vec2 curve[3] = {nvec2(0.046511628, 0.73975044), nvec2(0.09108527, 1.), nvec2(0.45639536, 1.)};
    vec2 tc = nvec2(0.25, 0.92);
    for (int i = 0; i< 3; i++) {
        curve[i] = sub(curve[i], tc);
    }
    vec2 roots = bezq_roots(curve);
    printf("%f, %f\n", roots.x, roots.y);
    vec2 pt = bezq_pt(curve, roots.x);
    vec2 pt2 = bezq_pt(curve, roots.y);
    printf("(%f, %f), (%f, %f)\n", pt.x, pt.y, pt2.x, pt2.y);
    
}