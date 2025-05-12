#pragma once
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// — Dimensiones GPU — 
constexpr int BSIZE    = 256;
constexpr int BLOCKS   = 16384 / BSIZE;

// — Dimensiones de la ventana —
constexpr unsigned int windowWidth  = 1280;
constexpr unsigned int windowHeight = 720;

// — Número de partículas —
constexpr int numBodies = 16384;

// — Recursos OpenGL/CUDA —
extern GLuint VAO, VBO;
extern struct cudaGraphicsResource* cudaVBO;
extern float4* d_particles;
extern float4* h_particles;

// — Parámetros de simulación —
extern int   gApprx;
extern int   gOffset;
extern float gStep;

// — Headers GPU y CPU —
extern "C"
void cudaComputeGalaxy(void* dptr,
                       float4* pdata,
                       int N,
                       float step,
                       int apprx,
                       int offset);

void runCuda(float time);

void loadDubinskiData(const std::string& path,
                      std::vector<float4>& positions,
                      std::vector<float4>& velocities);

GLuint compileShader(GLenum type, const char* src);
GLuint createShaderProgram();

void initGL(const std::vector<float4>& initialPositions,
            const std::vector<float4>& initialVelocities);
