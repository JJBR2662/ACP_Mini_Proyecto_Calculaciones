#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {
    string archivoCSV = "estudiantes.csv";
    vector<vector<double>> datos;
    ifstream archivo(archivoCSV);
    string linea, valor;

    getline(archivo, linea);

    while (getline(archivo, linea)) {
        stringstream ss(linea);
        vector<double> fila;
        bool primera_col = true;
        while (getline(ss, valor, ',')) {
            if (primera_col) {
                primera_col = false; // Omitir la primera columna (nombres de fila)
            }
            else {
                fila.push_back(stod(valor));
            }
        }
        datos.push_back(fila);
    }
    archivo.close();

    int filas = datos.size();
    int columnas = datos[0].size();
    MatrixXd X(filas, columnas);
    for (int i = 0; i < filas; ++i) {
        for (int j = 0; j < columnas; ++j) {
            X(i, j) = datos[i][j];
        }
    }
    cout << "Matriz original X:\n" << X << "\n\n";

    // Paso 1 Centrar y reducir la matriz
    MatrixXd X_norm = X;
    RowVectorXd media = X.colwise().mean();
    RowVectorXd desvest = ((X.rowwise() - media).array().square().colwise().mean()).sqrt();
    for (int i = 0; i < X.cols(); ++i) {
        X_norm.col(i) = (X.col(i).array() - media(i)) / desvest(i);
    }
    cout << "Matriz centrada y reducida:\n" << X_norm << "\n\n";

    // Paso 2: Calcular la matriz de correlación
    MatrixXd R = (X_norm.transpose() * X_norm) / (X_norm.rows() - 1);
    cout << "Matriz de correlación:\n" << R << "\n\n";

    // Paso 3: Calcular valores y vectores propios
    SelfAdjointEigenSolver<MatrixXd> eigensolver(R);
    VectorXd valoresPropios = eigensolver.eigenvalues().reverse();
    MatrixXd V = eigensolver.eigenvectors().rowwise().reverse();
    cout << "Valores propios:\n" << valoresPropios << "\n\n";
    cout << "Vectores propios (matriz V):\n" << V << "\n\n";

    // Paso 5: Calcular matriz de componentes principales
    MatrixXd C = X_norm * V;
    cout << "Matriz de componentes principales C:\n" << C << "\n\n";

    // Paso 6: Calcular la matriz de calidades de individuos
    MatrixXd Q(filas, columnas);
    for (int i = 0; i < filas; ++i) {
        for (int r = 0; r < columnas; ++r) {
            Q(i, r) = pow(C(i, r), 2) / X.row(i).array().square().sum();
        }
    }
    cout << "Matriz de calidades de individuos Q:\n" << Q << "\n\n";

    // Paso 7: Calcular la matriz de coordenadas de las variables
    MatrixXd T = V.transpose();
    cout << "Matriz de coordenadas de las variables T:\n" << T << "\n\n";

    // Paso 8: Calcular la matriz de calidades de las variables
    MatrixXd S = Q.transpose();
    cout << "Matriz de calidades de las variables S:\n" << S << "\n\n";

    // Paso 9: Calcular el vector de inercias de los ejes
    VectorXd I(columnas);
    for (int i = 0; i < columnas; ++i) {
        I(i) = 100.0 * valoresPropios(i) / columnas;
    }
    cout << "Vector de inercias de los ejes I:\n" << I << "\n";

    return 0;
}
