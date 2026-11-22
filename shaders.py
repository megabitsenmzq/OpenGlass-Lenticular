import gpu


VERTEX_SOURCE = """
void main() {
    uvInterp = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

COPY_FRAGMENT = """
void main() {
    fragColor = texture(image, uvInterp);
}
"""

DOWNSAMPLE_FRAGMENT = """
void main() {
    vec2 sourceSize = vec2(1440.0, 2560.0);
    vec2 scale = vec2(scale_x, scale_y);
    vec2 pixel = floor(uvInterp * sourceSize / scale);
    vec4 color = vec4(0.0);
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 4; y++) {
            if (x >= sample_x || y >= sample_y) {
                continue;
            }
            vec2 offset = (vec2(float(x), float(y)) + 0.5) * scale / vec2(float(sample_x), float(sample_y));
            color += texture(image, (pixel * scale + offset) / sourceSize);
        }
    }
    fragColor = color / float(sample_x * sample_y);
}
"""

INTERLACE_FRAGMENT = """
vec4 goSample(int view, vec2 localUV) {
    view = clamp(view, 0, view_count - 1);
    int chunk = view / (atlas_chunk_rows * view_columns);
    view -= chunk * atlas_chunk_rows * view_columns;
    int rows = min(atlas_chunk_rows, view_rows - chunk * atlas_chunk_rows);
    int column = view % view_columns;
    int row = view / view_columns;
    vec2 uv = (vec2(float(column), float(row)) + localUV) / vec2(float(view_columns), float(rows));
    if (chunk == 1) {
        return texture(image_upper, uv);
    }
    if (chunk == 2) {
        return texture(image_third, uv);
    }
    return texture(image, uv);
}

vec4 c1Choice(float bias) {
    float safeInterval = max(interval, 0.001);
    float x = floor(uvInterp.x * 1440.0) + 1.0;
    float y = floor((1.0 - uvInterp.y) * 2560.0) + 1.0;
    float phase = (x + y * slope) * 3.0 + bias + x0;
    float wrapped = phase - safeInterval * floor(phase / safeInterval);
    float continuous = clamp(wrapped / safeInterval, 0.0, 0.99999) * float(view_count);
    int choice = min(int(floor(continuous)), view_count - 1);
    int viewA = (view_count - 1) - choice;
    vec4 a = goSample(viewA, uvInterp);
    if (view_interpolation == 0) {
        return a;
    }
    int viewB = max(viewA - 1, 0);
    vec4 b = goSample(viewB, uvInterp);
    return mix(a, b, continuous - floor(continuous));
}

vec4 goChoice(int channel) {
    float subpixel = 1.0 / (3.0 * 1440.0);
    vec2 viewUV = vec2((floor(uvInterp.x * 1440.0) + 0.5) / 1440.0, uvInterp.y);
    float phase = (viewUV.x + float(channel) * subpixel + viewUV.y * slope) * interval - x0 + 0.5 / float(view_count);
    phase = 1.0 - fract(phase);
    phase = clamp(phase, 0.00001, 0.99999);
    float view = phase * float(view_count);
    int lower = min(int(floor(view)), view_count - 1);
    vec4 a = goSample(lower, uvInterp);
    if (view_interpolation == 0) {
        return a;
    }
    int upper = min(lower + 1, view_count - 1);
    vec4 b = goSample(upper, uvInterp);
    return mix(a, b, view - floor(view));
}

void main() {
    if (device_kind == 1) {
        vec4 color = goChoice(0);
        color.g = goChoice(1).g;
        color.b = goChoice(2).b;
        fragColor = vec4(color.rgb, 1.0);
        return;
    }
    vec4 color = c1Choice(0.0);
    color.g = c1Choice(1.0).g;
    color.b = c1Choice(2.0).b;
    fragColor = vec4(color.rgb, 1.0);
}
"""


def make_shaders():
    interface = gpu.types.GPUStageInterfaceInfo("lenticular_interface")
    interface.smooth("VEC2", "uvInterp")

    copy_info = gpu.types.GPUShaderCreateInfo()
    copy_info.vertex_in(0, "VEC2", "pos")
    copy_info.vertex_out(interface)
    copy_info.sampler(0, "FLOAT_2D", "image")
    copy_info.fragment_out(0, "VEC4", "fragColor")
    copy_info.vertex_source(VERTEX_SOURCE)
    copy_info.fragment_source(COPY_FRAGMENT)

    interlace_info = gpu.types.GPUShaderCreateInfo()
    interlace_info.vertex_in(0, "VEC2", "pos")
    interlace_info.vertex_out(interface)
    interlace_info.sampler(0, "FLOAT_2D", "image")
    interlace_info.sampler(1, "FLOAT_2D", "image_upper")
    interlace_info.sampler(2, "FLOAT_2D", "image_third")
    interlace_info.push_constant("FLOAT", "x0")
    interlace_info.push_constant("FLOAT", "interval")
    interlace_info.push_constant("FLOAT", "slope")
    interlace_info.push_constant("INT", "device_kind")
    interlace_info.push_constant("INT", "view_interpolation")
    interlace_info.push_constant("INT", "view_count")
    interlace_info.push_constant("INT", "view_columns")
    interlace_info.push_constant("INT", "view_rows")
    interlace_info.push_constant("INT", "atlas_chunk_rows")
    interlace_info.fragment_out(0, "VEC4", "fragColor")
    interlace_info.vertex_source(VERTEX_SOURCE)
    interlace_info.fragment_source(INTERLACE_FRAGMENT)

    return gpu.shader.create_from_info(copy_info), gpu.shader.create_from_info(interlace_info)


def make_downsample_shader():
    interface = gpu.types.GPUStageInterfaceInfo("lenticular_downsample_interface")
    interface.smooth("VEC2", "uvInterp")
    info = gpu.types.GPUShaderCreateInfo()
    info.vertex_in(0, "VEC2", "pos")
    info.vertex_out(interface)
    info.sampler(0, "FLOAT_2D", "image")
    info.push_constant("FLOAT", "scale_x")
    info.push_constant("FLOAT", "scale_y")
    info.push_constant("INT", "sample_x")
    info.push_constant("INT", "sample_y")
    info.fragment_out(0, "VEC4", "fragColor")
    info.vertex_source(VERTEX_SOURCE)
    info.fragment_source(DOWNSAMPLE_FRAGMENT)
    return gpu.shader.create_from_info(info)
