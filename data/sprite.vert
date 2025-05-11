   
void main()
{
	float pointSize = 500.0 * gl_Point.size;
	vec4 vert = gl_Vertex;
	vert.w = 1.0;
	vec3 pos_eye = vec3 (gl_ModelViewMatrix * vert);
	gl_PointSize = max(1.0, pointSize / (1.0 - pos_eye.z));
	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_Position = ftransform();
	gl_FrontColor = gl_Color;
}

