#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
layout(location = NORMAL_LOC) in vec3 normal;
layout(location = INST_M_LOC) in mat4 inst_m;
layout(location = COLOR_0_LOC) in vec4 color_0;

// Uniforms
uniform mat4 M; // model
uniform mat4 V; // view
uniform mat4 P; // projection

// Outputs
out vec3 frag_position;
out vec3 frag_normal;
out vec4 color_multiplier;

void main()
{
    gl_Position = P * V * M * inst_m * vec4(position, 1);
    frag_position = vec3(M * inst_m * vec4(position, 1.0));

    // original, for some reason
    //mat4 N = transpose(inverse(M * inst_m));
    //frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
    mat4 N = M * inst_m;
    frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
    color_multiplier = color_0;
}