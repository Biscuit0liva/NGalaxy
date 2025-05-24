#include "shared.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iomanip>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#include <GL/glx.h>

// OpenCL
#include <CL/cl.h>
using float4 = cl_float4;
#include <CL/cl_gl.h>

// Variables para calcular FPS
auto lastTime = std::chrono::high_resolution_clock::now();
int frameCount = 0;

// Archivo de resultados
std::ofstream resultsFile;

// Recursos generales
GLuint VAO, VBO;
GLuint shaderProgram;
float4* h_particles = nullptr;

// Recursos OpenCL
cl_context clContext;
cl_command_queue clQueue;
cl_program clProgram;
cl_kernel clKernel;
cl_mem clInteropBuffer;
cl_mem clDataBuffer;

// Parámetros de simulación
int gApprx = 1;
int gOffset = 0;
float gStep = 0.001f;

void checkCLErr(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error: " << msg << " (" << err << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

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

        float4 p{ vals[1]*1.5f, vals[2]*1.5f, vals[3]*1.5f, vals[0]*120000.0f };

        float4 v{ vals[4]*8.0f, vals[5]*8.0f, vals[6]*8.0f, 1.0f };

        
        h_particles[i] = p;
        h_particles[i + numBodies] = v;

        positions.push_back({ p.s[0], p.s[1], p.s[2], p.s[3] });
        velocities.push_back({ v.s[0], v.s[1], v.s[2], v.s[3] });
        ++i;
    }
}

static GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512];
        glGetShaderInfoLog(shader, 512, nullptr, buf);
        std::cerr << "Shader compile error: " << buf << std::endl;
        std::exit(1);
    }
    return shader;
}

GLuint createShaderProgram() {
    const char* vsSrc = R"(
    #version 330 core
    layout(location = 0) in vec4 position;
    layout(location = 1) in vec4 velocity;
    out vec3 vColor;

    void main() {
        gl_PointSize = 2.5;
        gl_Position = vec4(position.xyz * 0.02, 1.0);
        
        vec3 dir = normalize(velocity.xyz);
        vColor = 0.5 + 0.5 * dir;
    }
    )";
    const char* fsSrc = R"(
    #version 330 core
    in vec3 vColor;
    out vec4 FragColor;

    void main() {
        FragColor = vec4(vColor, 1.0);
    }
    )";
    GLuint vs = compileShader(GL_VERTEX_SHADER,   vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

void buildOpenCL(const std::string& path) {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    checkCLErr(clGetPlatformIDs(1, &platform, nullptr), "get platform");
    checkCLErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "get device");

    size_t sz=0;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &sz);
    std::string exts(sz, '\0');
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sz, &exts[0], nullptr);
    std::cout << "OpenCL_EXTS: " << exts << "\n";


    Display* x11Display = glXGetCurrentDisplay();
    GLXContext glContext = glXGetCurrentContext();

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM,    (cl_context_properties)platform,
        CL_GL_CONTEXT_KHR,      (cl_context_properties)glContext,
        CL_GLX_DISPLAY_KHR,     (cl_context_properties)x11Display,
        0
    };
    assert(glXGetCurrentContext() == glContext && "No hay contexto GL actual!");
    clContext = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    checkCLErr(err, "create context");
    clQueue = clCreateCommandQueue(clContext, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkCLErr(err, "create queue");

    std::ifstream file(path);
    std::string src((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const char* srcPtr = src.c_str();
    size_t srcSize = src.length();

    clProgram = clCreateProgramWithSource(clContext, 1, &srcPtr, &srcSize, &err);
    checkCLErr(err, "create program");

    err = clBuildProgram(clProgram, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[2048];
        clGetProgramBuildInfo(clProgram, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "OpenCL build log:\n" << log << "\n";
        std::exit(EXIT_FAILURE);
    }

    clKernel = clCreateKernel(clProgram, "galaxyKernel", &err);
    checkCLErr(err, "create kernel");
}

void runOpenCL(float time) {
    cl_int err;

    glFinish();  // asegura que GL haya terminado

    checkCLErr(clEnqueueAcquireGLObjects(clQueue, 1, &clInteropBuffer, 0, nullptr, nullptr), "acquire GL buffer");

    gOffset = (gOffset + 1) % gApprx;

    clSetKernelArg(clKernel, 0, sizeof(cl_mem), &clInteropBuffer);
    clSetKernelArg(clKernel, 1, sizeof(cl_mem), &clDataBuffer);
    clSetKernelArg(clKernel, 2, sizeof(int), &numBodies);
    clSetKernelArg(clKernel, 3, sizeof(float), &gStep);
    clSetKernelArg(clKernel, 4, sizeof(int), &gApprx);
    clSetKernelArg(clKernel, 5, sizeof(int), &gOffset);

    size_t global = numBodies;
    size_t local  = BSIZE;
    cl_event kernelEvent;

    checkCLErr(clEnqueueNDRangeKernel(clQueue, clKernel, 1, nullptr, &global, &local, 0, nullptr, &kernelEvent), "enqueue kernel");

    clFinish(clQueue);

    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(kernelEvent,
                            CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong),
                            &t0,
                            nullptr);
    clGetEventProfilingInfo(kernelEvent,
                            CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong),
                            &t1,
                            nullptr);

    double ms = (double)(t1 - t0) * 1e-6;

    double secs = ms / 1000.0;
    long long totalInteractions = (long long)numBodies * numBodies;
    double interactionsPerSec = totalInteractions / secs;

    // Metricas
    std::cout << "Paso: " << time
              << " | Tiempo kernel: " << ms << " ms"
              << " | Interacciones/s: "   << interactionsPerSec
              << std::endl;

    // registro en archivo
    resultsFile << std::fixed << std::setprecision(3) << ms << ","
        << std::setprecision(0) << interactionsPerSec << "\n";   

    // Liberar recursos
    clReleaseEvent(kernelEvent);
    checkCLErr(
      clEnqueueReleaseGLObjects(clQueue, 1, &clInteropBuffer, 0, nullptr, nullptr),
      "release GL buffer");
}

void initGLCL(const std::vector<float4>& positions, const std::vector<float4>& velocities) {
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

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 2 * sizeof(float4), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 2 * sizeof(float4), (void*)(sizeof(float4)));

    cl_int err;
    clInteropBuffer = clCreateFromGLBuffer(clContext, CL_MEM_WRITE_ONLY, VBO, &err);
    checkCLErr(err, "create interop buffer");

    clDataBuffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float4) * numBodies * 2, h_particles, &err);
    checkCLErr(err, "create data buffer");
}

int main() {
    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Galaxy OpenCL", nullptr, nullptr);
    if (!window) {
        std::cerr << "GLFW window creation failed\n";
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }

    std::vector<float4> pos, vel;
    loadDubinskiData("data/dubinski.tab", pos, vel);

    buildOpenCL("build/galaxy_kernel.cl");
    shaderProgram = createShaderProgram();
    initGLCL(pos, vel);
    glEnable(GL_PROGRAM_POINT_SIZE);

    float time = 0.0f;
    std::string experimentName = "opencl_global";
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
    while (!glfwWindowShouldClose(window)) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsedSec = std::chrono::duration<float>(now - benchmarkStart).count();
        if (elapsedSec >= 30.0f) break;

        glfwPollEvents();
        runOpenCL(time);

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

    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(shaderProgram);
    clReleaseMemObject(clInteropBuffer);
    clReleaseMemObject(clDataBuffer);
    clReleaseKernel(clKernel);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(clQueue);
    clReleaseContext(clContext);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
