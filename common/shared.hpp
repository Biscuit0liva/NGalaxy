#pragma once


// — Configuración de la simulación — 
constexpr int    BSIZE       = 248;                 // tamaño de bloque Probamos con 256, 248 y 231
constexpr int    numBodies   = 32768;             // partículas totales
constexpr int    BLOCKS      = numBodies / BSIZE;       // número de bloques

// — Dimensiones de la ventana — 
constexpr unsigned int windowWidth  = 1280;
constexpr unsigned int windowHeight = 720;

