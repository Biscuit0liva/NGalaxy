#pragma once


// — Configuración de la simulación — 
constexpr int    BSIZE       = 256;                 // tamaño de bloque Probamos con 256, 248 y 231
constexpr int    BLOCKS      = 16384 / BSIZE;       // número de bloques
constexpr int    numBodies   = 16384;               // partículas totales

// — Dimensiones de la ventana — 
constexpr unsigned int windowWidth  = 1280;
constexpr unsigned int windowHeight = 720;

