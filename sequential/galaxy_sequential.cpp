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

#include <chrono> // Para medir el tiempo

// Variables para calcular FPS
auto lastTime = std::chrono::high_resolution_clock::now();
int frameCount = 0;

struct float4 {
    float x, y, z, w;
};

struct float3 {
    float x, y, z;
};

// Recursos generales
GLuint VAO, VBO;
GLuint shaderProgram;
std::vector<float4> particles; 

// Parámetros de simulación
int gApprx = 64;
int gOffset = 0;
float gStep = 0.001f;



// ==========================================================
// CPU Physics
// ==========================================================
void computeGalaxyCPU(std::vector<float4>& particles, int N, float step, int apprx, int offset){
    const float softeningSquared = 0.01f;
    const float damping = 1.0f;
    const float ep = 0.67f;

    std::vector<float4> newPos(N);
    std::vector<float4> newVel(N);

    int start = (N / apprx) * offset;

    for(int i=0; i<N; i++){
        float4 pi = particles[i];
        float4 vi = particles[i + N];

        float3 acc = {0.0f, 0.0f, 0.0f};

        for (int j = start; j < start + (N / apprx); j++) {
            float4 pj = particles[j];

            float3 r = {pj.x - pi.x, pj.y - pi.y, pj.z - pi.z};
            float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + softeningSquared;
            float dist = sqrtf(distSqr);
            float distCube = dist * dist * dist;

            if (distCube < 1.0f) continue;

            float s = pj.w / distCube;
            acc.x += r.x * s * ep;
            acc.y += r.y * s * ep;
            acc.z += r.z * s * ep;
        }

        vi.x += acc.x * step;
        vi.y += acc.y * step;
        vi.z += acc.z * step;

        vi.x *= damping;
        vi.y *= damping;
        vi.z *= damping;

        pi.x += vi.x * step;
        pi.y += vi.y * step;
        pi.z += vi.z * step;
        
        newPos[i] = pi;
        newVel[i] = vi;
    }

    for(int i=0; i<N; i++){
        particles[i] = newPos[i];
        particles[i + N] = newVel[i];
    }
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

        positions.push_back(p);
        velocities.push_back(v);
        i++;
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
void initGL(std::vector<float4>& positions, std::vector<float4>& velocities) {
    std::vector<float4> interleaved;
    interleaved.reserve(numBodies * 2);
    for (int i = 0; i < numBodies; ++i) {
        interleaved.push_back(positions[i]);
        interleaved.push_back(velocities[i]);
    }
    particles = interleaved;

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, interleaved.size() * sizeof(float4), interleaved.data(), GL_DYNAMIC_DRAW);
    
    // posicion -> location =0
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 2 * sizeof(float4), (void*)0);

    // velocidad -> location =1
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 2 * sizeof(float4), (void*)(sizeof(float4)));
}

// ==========================================================
int main() {
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Galaxy CPU", nullptr, nullptr);
    if (!window) {
        std::cerr << "GLFW window creation failed\n";
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }

    std::vector<float4> pos, vel;
    loadDubinskiData("data/dubinski.tab", pos, vel);
    shaderProgram = createShaderProgram();
    initGL(pos, vel);
    glEnable(GL_PROGRAM_POINT_SIZE);

    float time = 0.0f;
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        gOffset = (gOffset + 1) % gApprx;
        auto start = std::chrono::high_resolution_clock::now();
        computeGalaxyCPU(particles, numBodies, gStep, gApprx, gOffset);
        auto stop = std::chrono::high_resolution_clock::now();

        float ms = std::chrono::duration<float, std::milli>(stop - start).count();
        double s = ms / 1000.0;
        double ips = (double)numBodies * numBodies / s;
        std::cout << "Paso: " << time << " | Tiempo CPU: " << ms << " ms | Interacciones/s: " << ips << std::endl;

        // copiar datos actualizados al VBO
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(float4), particles.data());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, numBodies);
        glfwSwapBuffers(window);
        // Calcular FPS
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

    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
