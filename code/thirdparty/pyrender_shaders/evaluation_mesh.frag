#version 330 core

in vec3 frag_position;
in vec3 frag_normal;
in vec4 color_multiplier;

out vec4 frag_color;

void main()
{
    frag_color = color_multiplier;
}