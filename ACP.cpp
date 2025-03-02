#include "ACP.h"
#include <iostream>
#include <Eigen/Eigenvalues>  // Para cálculo de valores y vectores propios

ACP::ACP(const Eigen::MatrixXd& data) : X(data) {}

void ACP::calcularACP() {
    // Calculamos la matriz de correlación
    Eigen::MatrixXd R = (X.transpose() * X) / double(X.rows() - 1);

    // Descomposición en valores propios
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(R);

    // se guarda los valores propios en la variable de la clase
    valoresPropios = solver.eigenvalues().reverse();

    // guarda los vectores propios
    V = solver.eigenvectors().rowwise().reverse();

    // Matriz de componentes principales
    C = X * V;
}

void ACP::calcularMatrizCalidadIndividuos() {
    int n = X.rows();  // Número de individuos
    int m = X.cols();  // Número de variables

    Q = Eigen::MatrixXd(n, m);

    // Calcular el denominador para cada uno 
    Eigen::VectorXd denominador = X.array().square().rowwise().sum();

    // Evitar división por cero
    for (int i = 0; i < n; ++i) {
        if (denominador(i) == 0) {
            denominador(i) = 1e-8;
        }
    }

    // Calcular Q 
    for (int i = 0; i < n; ++i) {
        for (int r = 0; r < m; ++r) {
            Q(i, r) = std::pow(C(i, r), 2) / denominador(i);
        }
    }
}

void ACP::calcularMatrizCoordenadasVariables() {
    T = V.transpose() * X.transpose();
}

void ACP::calcularMatrizCalidadVariables() {
    S = T.array().square();
}

void ACP::calcularVectorInercias() {
    int m = X.cols();  // Número de variables

    // condicional para que I sea un vector fila de tamaño 1xm
    I = Eigen::RowVectorXd(m);

    for (int j = 0; j < m; ++j) {
        I(j) = 100.0 * valoresPropios(j) / m;
    }
}

void ACP::mostrarResultados() {
    std::cout << "Matriz de Componentes Principales (C):\n" << C << "\n\n";
    std::cout << "Matriz de Calidades de Individuos (Q):\n" << Q << "\n\n";
    std::cout << "Matriz de Coordenadas de Variables (T):\n" << T << "\n\n";
    std::cout << "Matriz de Calidades de Variables (S):\n" << S << "\n\n";
    std::cout << "Vector de Inercias (I):\n" << I.transpose() << "\n\n";
}
