#version 450

struct GlyphInfo{
    uint offset;
    uint len;
};
layout(location = 0) in uint index;
layout(location = 0) out vec2 tex_coords;
layout(location = 1) out uint offset;
layout(location = 2) out uint len;
layout(set = 0, binding = 1) buffer font_meta{
    GlyphInfo[] glyphs;
};
layout(set = 0, binding = 2) buffer page{
    uint[] chars;
};
layout (push_constant) uniform constants {
    ivec2 resolution;
};
#define PI 3.14159
const vec2[6] vectors = vec2[](vec2(0., 0.), vec2(0., 1.), vec2(1., 1.), vec2(0., 0.), vec2(1., 0.), vec2(1., 1.));
void main() {
    int num_cols = resolution.x / 20;
    int num_rows = resolution.y / 30;
    float width = ((float(num_cols * 20) / float(resolution.x)) *2. )/float(num_cols);
    float height = ((float(num_rows * 30) / float(resolution.y)) *2. )/float(num_rows);
    uint char_offset = index / 6;
    GlyphInfo info = glyphs[chars[char_offset]];
    offset = info.offset;
    len = info.len;
    uint row = char_offset / num_cols;
    uint col = char_offset % num_cols;
    vec2 quad_offset = vec2(float(col), float(row))*vec2(width, height);
    uint vec_index = index % 6;
    vec2 position = vec2(-1., -1.) + quad_offset + (vectors[vec_index] * vec2(width, height));
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = vectors[vec_index];
}

