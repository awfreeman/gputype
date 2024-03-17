#version 450

struct GlyphInfo{
    uint offset;
    uint len;
};
layout(location = 0) in uint index;
layout(location = 0) out vec2 tex_coords;
layout(location = 1) out uint offset;
layout(location = 2) out uint len;
layout(set = 0, binding = 1) buffer glyph_info{
    GlyphInfo[] glyphs;
};
#define PI 3.14159
const vec2[6] vectors = vec2[](vec2(0., 0.), vec2(0., 1.), vec2(1., 1.), vec2(0., 0.), vec2(1., 0.), vec2(1., 1.));
const vec2[6] tex_vectors = vec2[](vec2(0., 1.), vec2(1., 1.), vec2(1., 0.), vec2(0., 1.), vec2(0., 0.), vec2(1., 0.));
void main() {
    uint which = index/6;
    float char_pos = float(which);
    uint vec_index = index % 6;
    vec2 root_offset = vec2(-1., 0.) + (char_pos*vec2(0.1, 0.0));
    vec2 position = root_offset + (vectors[vec_index] * 0.1);
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = vectors[vec_index];
    offset = glyphs[which].offset;
    len = glyphs[which].len;
}