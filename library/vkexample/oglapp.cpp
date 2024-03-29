#include "oglapp.h"
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

float grey = 0.0f;

#ifdef OPENGL_3_APP
const char* vertex_shader_code =
"#version 330 core\n"
"layout(location = 0) in vec3 aPos;\n"
"void main() {\n"
"    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";

const char* fragment_shader_code = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main() {\n"
"    FragColor = vec4(1.0, 0.5, 0.2, 1.0);\n"
"}\0";
#else
const char* vertex_shader_code =
"#version 110\n"
"attribute vec3 aPos;\n"
"void main() {\n"
"    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";

const char* fragment_shader_code = "#version 110\n"
"void main() {\n"
"    gl_FragColor = vec4(1.0, 0.5, 0.2, 1.0);\n"
"}\0";
#endif

int w = 1280, h = 720;

GLuint program;
GLuint vao, vbo;

int createShader(const char* code, int type) {
    int shader;
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &code, NULL);
    glCompileShader(shader);
    int status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if(!status) {
        char info[512];
        glGetShaderInfoLog(shader, 512, NULL, info);
        printf("shader error: %s\n", info);
    }
    return shader;
}

void initialize() {
    // create shaders
    int vertex_shader = createShader(vertex_shader_code, GL_VERTEX_SHADER);
    int fragment_shader = createShader(fragment_shader_code, GL_FRAGMENT_SHADER);

    program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    int status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if(!status) {
        char info[512];
        glGetProgramInfoLog(program, 512, NULL, info);
        printf("shader error: %s\n", info);
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.0f, 0.5f, 0.0f
    };
#ifdef OPENGL_3_APP
    glGenVertexArrays(1, &vao);
#endif

    glGenBuffers(1, &vbo);
#ifdef OPENGL_3_APP
    glBindVertexArray(vao);
#endif
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(float), vertices, GL_STATIC_DRAW);

    // load data
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
#ifdef OPENGL_3_APP
    glBindVertexArray(0);
#endif
}

void render(float deltaTime) {
    glClearColor(grey, grey, grey, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);

#ifdef OPENGL_3_APP
    glBindVertexArray(vao);
#else
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
#endif
    glDrawArrays(GL_TRIANGLES, 0, 3);

    grey += deltaTime/5.f;
    if(grey > 1.0f) {
        grey = 0.0f;
    }
}

void glfwError(int i, const char* des) {
    printf("%s\n", des);
}

void run_opengl_app() {
    const char* name = "super opengl example";
    glfwSetErrorCallback(&glfwError);
    if(!glfwInit())
    {
        printf("failed to initialize GLFW");
        return;
    }

#ifdef OPENGL_3_APP
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif
    

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(w, h, name, NULL, NULL);
    if(!window)
    {
        printf("failed to create GLFW window %d %d\n", w, h);
        return;
    }
    glfwMakeContextCurrent(window);

    if(!gladLoadGL(glfwGetProcAddress)) {
        printf("error loading opengl\n");
        glfwTerminate();
        return;
    }

    float lastTime = (float)glfwGetTime();

    const GLubyte* vendor = glGetString(GL_VENDOR);
    const GLubyte* device = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    printf("OpenGL Vendor: %s\nOpenGL Device: %s\nOpenGL Version: %s\n", vendor, device, version);

    initialize();

    float accumTime = 0.0f;
    uint32_t accumFrames = 0;

    glViewport(0, 0, w, h);

    while(!glfwWindowShouldClose(window))
    {
        float curTime = (float)glfwGetTime();
        float dt = curTime - lastTime;
        lastTime = curTime;

        accumTime += dt;
        accumFrames++;

        if(accumTime >= 1.0f)
        {
            float avgDt = accumTime / accumFrames;

            char windowName[64];
            snprintf(windowName, sizeof(windowName), "super opengl example [FPS: %.0f (%.2fms)]", 1.0f / avgDt, avgDt * 1000.0f);
            glfwSetWindowTitle(window, windowName);

            accumTime -= 1.0f;
            accumFrames = 0;
        }

        render(dt);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
}