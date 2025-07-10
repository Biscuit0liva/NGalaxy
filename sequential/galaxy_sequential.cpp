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
GLuint VAO;
GLuint VBO;
GLuint shaderProgram;
float4* h_particles = nullptr;

std::vector<float4> particles;
float4* dptr;

// Parámetros de simulación
int gApprx = 1;
int gOffset = 0;
float gStep = 0.001f;



// ==========================================================
// CPU Physics
// ==========================================================
void computeGalaxyCPU(float4* dptr, std::vector<float4>& particles, int N, float step, int apprx, int offset){
    const float softeningSquared = 0.01f;
    const float damping = 1.0f;
    const float ep = 0.67f;
    std::vector<float4> newParticles(N * 2);
    for(int i=0; i<N; i++){
        unsigned int pLoc = i;
        unsigned int vLoc = i + N;

        float4 myPosition = particles[pLoc];
        float4 myVelocity = particles[vLoc];

        float3 acc = {0.0f, 0.0f, 0.0f};
        float3 r;
        float distSqr, distCube, s;

        for(int j=0; j<N; j++){
            r.x = particles[j].x - myPosition.x;
            r.y = particles[j].y - myPosition.y;
            r.z = particles[j].z - myPosition.z;

            distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
            distSqr += softeningSquared;

            float dist = sqrtf(distSqr);
            distCube = dist * dist * dist;
            //if (distCube < 1.0f) continue;

            s = particles[j].w / distCube;

            acc.x += r.x * s * ep;
            acc.y += r.y * s * ep;
            acc.z += r.z * s * ep;
        }

        myVelocity.x += acc.x * step;
        myVelocity.y += acc.y * step;
        myVelocity.z += acc.z * step;

        myVelocity.x *= damping;
        myVelocity.y *= damping;
        myVelocity.z *= damping;

        myPosition.x += myVelocity.x * step;
        myPosition.y += myVelocity.y * step;
        myPosition.z += myVelocity.z * step;

        // Guardar resultados
        newParticles[pLoc] = myPosition;
        newParticles[vLoc] = myVelocity;

        dptr[2 * pLoc] = float4{myPosition.x, myPosition.y, myPosition.z, 1.0f};
        dptr[2 * pLoc+1] = float4{myVelocity.x, myVelocity.y, myVelocity.z, 1.0f};
    }
    // Actualizar partículas
    for(int i=0; i<N; i++){
        particles[i] = newParticles[i];
        particles[i + N] = newParticles[i + N];
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
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    std::vector<float4> interleaved;
    interleaved.reserve(numBodies * 2);
    for (int i = 0; i < numBodies; ++i) {
        interleaved.push_back(positions[i]);
        interleaved.push_back(velocities[i]);
    }

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
    particles.assign(h_particles, h_particles + numBodies * 2);

    shaderProgram = createShaderProgram();
    initGL(pos, vel);
    glEnable(GL_PROGRAM_POINT_SIZE);
    

    float time = 0.0f;
    auto lastTime = std::chrono::high_resolution_clock::now();
    auto benchmarkStart = std::chrono::high_resolution_clock::now();
    int currentBuffer = 0;
    while (!glfwWindowShouldClose(window)) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsedSec = std::chrono::duration<float>(now - benchmarkStart).count();
        if (elapsedSec >= 30.0f) break;

        glfwPollEvents();

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        dptr = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        if (!dptr) {
            std::cerr << "Error: no se pudo mapear el VBO para escritura." << std::endl;
            break;
        }

        gOffset = (gOffset + 1) % gApprx;
        auto start = std::chrono::high_resolution_clock::now();
        computeGalaxyCPU(dptr, particles, numBodies, gStep, gApprx, gOffset);
        auto stop = std::chrono::high_resolution_clock::now();

        glUnmapBuffer(GL_ARRAY_BUFFER);

        float ms = std::chrono::duration<float, std::milli>(stop - start).count();
        double s = ms / 1000.0;
        double ips = (double)numBodies * numBodies / s;
        std::cout << "Paso: " << time << " | Tiempo CPU: " << ms << " ms | Interacciones/s: " << ips << std::endl;
        

        // copiar datos actualizados al VBO: TODO
        std::cout << "Actualizando VBO con " << numBodies * 2 << " partículas." << std::endl;
        std::cout << "Actualizando 2 VBO con " << numBodies * 2 << " partículas." << std::endl;
        
        // Usar el de lectura para visualizacion
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 2 * sizeof(float4), (void*)0);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 2 * sizeof(float4), (void*)(sizeof(float4)));
        std::cout << "Actualizando 3 VBO con " << numBodies * 2 << " partículas." << std::endl;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, numBodies);
        glfwSwapBuffers(window);
        // alternar buffers
        currentBuffer = 1 - currentBuffer;
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
