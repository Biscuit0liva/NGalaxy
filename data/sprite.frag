
uniform sampler2D splatTexture;
uniform int mode;
        
void main()
{
	if (mode == 1)
	{
		vec4 color = (0.1 + 0.9 * gl_Color) * texture2D(splatTexture, gl_TexCoord[0].st);
		gl_FragColor = color;
	}
	else
	{
		vec4 color = (0.6 + 0.4 * gl_Color) * texture2D(splatTexture, gl_TexCoord[0].st);
		gl_FragColor = color * mix( vec4(0.1, 0.0, 0.0, color.w), 
									vec4(1.0, 0.7, 0.3, color.w), 
									color.w );
	}
}
