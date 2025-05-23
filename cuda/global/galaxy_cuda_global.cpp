// same code as galaxy_cuda.cpp except for the kernel function, this one use the global memory verison of the kernel
#include "shared.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// CUDA 
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <chrono> // Para medir el tiempo
# include <iomanip> // Para formatear la salida

// Variables para calcular FPS
auto lastTime = std::chrono::high_resolution_clock::now();
int frameCount = 0;
// archivo de resultados
std::ofstream resultsFile;

// Recursos generales
GLuint VAO, VBO;
GLuint shaderProgram;
float4* h_particles = nullptr;

// Recursos CUDA
struct cudaGraphicsResource* cudaVBO = nullptr;
float4* d_particles = nullptr;


// no usar aproximado
// Parámetros de simulación
int gApprx = 1;
int gOffset = 0;
float gStep = 0.001f;


// ==========================================================
// CUDA kernel function to compute galaxy simulation
// ==========================================================
extern "C" void cudaComputeGalaxyGlobal(void* dptr, float4* pdata, int N, float step, int apprx, int offset);


// ==========================================================
void runCuda(float time) {
    float4* dptr;
    size_t num;
    cudaGraphicsMapResources(1, &cudaVBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num, cudaVBO);

    gOffset = (gOffset + 1) % gApprx;
    // Medicion de tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // Ejecuta el kernel
    cudaComputeGalaxyGlobal(dptr, d_particles, numBodies, gStep, gApprx, gOffset);
    // Calcula el tiempo de ejecución
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // evaluacion del rendimiento
    double seconds = milliseconds / 1000.0;
    long long totalInteractions = static_cast<long long>(numBodies) * numBodies;
    double interactionsPerSec = totalInteractions / seconds;
    // registro en consola y archivo
    std::cout << "Paso: " << time
                << " | Tiempo kernel: " << milliseconds << " ms"
                << " | Interacciones/s: " << interactionsPerSec << std::endl;
    
    resultsFile << std::fixed << std::setprecision(3) << milliseconds << ","
                << std::setprecision(0) << interactionsPerSec << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaGraphicsUnmapResources(1, &cudaVBO, 0);
}

// ==========================================================
void loadDubinskiData(const std::string& path, std::vector<float4>& positions, std::vector<float4>& velocities) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << "\n";
        exit(1);
    }

    int skip = 49152 / numBodies;
    std::string line;
    float vals[7];
    int count = 0;

    h_particles = new float4[numBodies * 2];

    for (int i = 0; i < numBodies && std::getline(file, line); ) {
        for (int s = 1; s < skip && std::getline(file, line); s++); // skip

        std::istringstream ss(line);
		for (int j = 0; j < 7; j++) {
    		if (!(ss >> vals[j])) {
        		std::cerr << "Failed to parse line: " << line << std::endl;
        		exit(1);
    	    }
        }

        float4 p, v;
        p.x = vals[1] * 1.5f;
        p.y = vals[2] * 1.5f;
        p.z = vals[3] * 1.5f;
        p.w = vals[0] * 120000.0f;

        v.x = vals[4] * 8.0f;
        v.y = vals[5] * 8.0f;
        v.z = vals[6] * 8.0f;
        v.w = 1.0f;
        
        h_particles[i] = p;
        h_particles[i + numBodies] = v;

        positions.push_back({p.x, p.y, p.z, 1.0f});
        velocities.push_back({v.x, v.y, v.z, 1.0f});
        ++i;
    }
}

// ==========================================================
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char msg[512];
        glGetShaderInfoLog(shader, 512, nullptr, msg);
        std::cerr << "Shader compile error: " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
    return shader;
}

GLuint createShaderProgram() {
    const char* vertexSrc = R"(
    #version 330 core
    layout(location = 0) in vec4 position;
    layout(location = 1) in vec4 velocity;

    out vec3 vColor;

    void main() {
        gl_PointSize = 2.5;
        gl_Position = vec4(position.xyz * 0.02, 1.0);

        // Codificación de color basada en la dirección de la velocidad
        vec3 dir = normalize(velocity.xyz);
        vColor = 0.5 + 0.5 * dir; // Rango [0,1]
    })";

    const char* fragmentSrc = R"(
        #version 330 core
        in vec3 vColor;
        out vec4 FragColor;

        void main() {
            FragColor = vec4(vColor, 1.0);
        })";

    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

// ==========================================================
void initGL(std::vector<float4>& initialPositions, std::vector<float4>& initialVelocities) {
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    // Crea un buffer intercalado: [posiciones, velocidades] 
    std::vector<float4> interleaved;
    interleaved.reserve(numBodies * 2);
    for (int i = 0; i < numBodies; ++i) {
        interleaved.push_back(initialPositions[i]);
        interleaved.push_back(initialVelocities[i]);
    }

    glBufferData(GL_ARRAY_BUFFER, interleaved.size() * sizeof(float4), interleaved.data(), GL_DYNAMIC_DRAW);
    
    // posicion -> location =0
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 2 * sizeof(float4), (void*)0);

    // velocidad -> location =1
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 2 * sizeof(float4), (void*)(sizeof(float4)));

    cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO, cudaGraphicsMapFlagsWriteDiscard);
}

// ==========================================================
int main() {
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Galaxy CUDA", nullptr, nullptr);
    if (!window) {
        std::cerr << "GLFW window creation failed\n";
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }

    cudaSetDevice(0);

    std::vector<float4> initialPositions, initialVelocities;
    loadDubinskiData("data/dubinski.tab", initialPositions, initialVelocities);

    cudaMalloc(&d_particles, sizeof(float4) * numBodies * 2);
    cudaMemcpy(d_particles, h_particles, sizeof(float4) * numBodies * 2, cudaMemcpyHostToDevice);

    shaderProgram = createShaderProgram();
    initGL(initialPositions, initialVelocities);
    glEnable(GL_PROGRAM_POINT_SIZE);

    float time = 0.0f;
    // Archivo para guardar resultados
    std::string experimentName = "cuda_global";
    // El nombre contiene: version de la implementacion, numero de particulas, tamaño de bloque
    std::string fileName = "results_"+experimentName+"_"+std::to_string(numBodies)+"_"+std::to_string(BSIZE)+".csv";
    resultsFile.open(fileName);
    if(!resultsFile.is_open()){
        std::cerr << "No se pudo abrir el archivo de resultados:" << fileName << std::endl;
        return -1;
    }
    resultsFile << "time_ms, interactions_per_sec\n" << std::endl;
    // Ejecuta la simulacion por 10 segundos
    auto benchmarkStart = std::chrono::high_resolution_clock::now();
    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsedSec = std::chrono::duration<float>(now - benchmarkStart).count();
        if (elapsedSec >= 10.0f) break;

        glfwPollEvents();
        runCuda(time);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, numBodies);
        glfwSwapBuffers(window);
        
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsedTime = currentTime - lastTime;
        if (elapsedTime.count() >= 1.0f) { // Cada segundo
            float fps = frameCount / elapsedTime.count();
            std::cout << "FPS: " << fps << std::endl;
            frameCount = 0;
            lastTime = currentTime;
        }
        time += 1.0f;
    }

    cudaGraphicsUnregisterResource(cudaVBO);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    resultsFile.close();
    return 0;
}
