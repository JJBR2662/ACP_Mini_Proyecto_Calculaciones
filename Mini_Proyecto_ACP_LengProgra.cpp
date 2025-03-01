#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "ACP.h"
using namespace std;


int main() {
    string archivo = "datos.csv";

    // Abrir el archivo en modo lectura
    ifstream archivo_csv(archivo);
    if (!archivo_csv.is_open()) {
        cerr << "No se pudo abrir el archivo." << endl;
        return 1;
    }

    // Leer la primera línea para determinar el número de columnas
    string linea;
    getline(archivo_csv, linea);
    stringstream ss(linea);
    string valor;
    int num_columnas = 0;
    while (getline(ss, valor, ',')) {
        ++num_columnas;
    }

    // Volver al principio del archivo
    archivo_csv.clear();
    archivo_csv.seekg(0, ios::beg);

    // Leer el archivo y contar el número de filas
    int num_filas = 0;
    while (getline(archivo_csv, linea)) {
        ++num_filas;
    }

    // Volver al principio del archivo
    archivo_csv.clear();
    archivo_csv.seekg(0, ios::beg);

    // Crear una matriz de Eigen con el tamaño adecuado
    Eigen::MatrixXd matriz(num_filas, num_columnas);

    // Leer los valores del archivo y almacenarlos en la matriz
    int fila = 0;
    while (getline(archivo_csv, linea)) {
        stringstream ss(linea);
        int columna = 0;
        while (getline(ss, valor, ',')) {
            matriz(fila, columna) = stod(valor); // Convertir a double
            ++columna;
        }
        ++fila;
    }

    archivo_csv.close(); // Cerrar el archivo

    // 1. Centrar y reducir la matriz
    Eigen::VectorXd media = matriz.colwise().mean();
    Eigen::VectorXd desviacion = ((matriz.rowwise() - media.transpose()).array().square().colwise().sum() / (matriz.rows() - 1)).sqrt();
    Eigen::MatrixXd matriz_estandarizada = (matriz.rowwise() - media.transpose()).array().rowwise() / desviacion.transpose().array();

    // 2. Calcular la matriz de correlaciones
    Eigen::MatrixXd matriz_correlaciones = matriz_estandarizada.transpose() * matriz_estandarizada / double(matriz_estandarizada.rows() - 1);

    // 3. Calcular y ordenar los vectores y valores propios
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matriz_correlaciones);
    if (eigensolver.info() != Eigen::Success) {
        cerr << "Error en la descomposición espectral." << endl;
        return 1;
    }
    Eigen::VectorXd valores_propios = eigensolver.eigenvalues().real();
    Eigen::MatrixXd vectores_propios = eigensolver.eigenvectors().real();

    // Ordenar los vectores y valores propios en orden descendente de los valores propios
    vector<int> indices(valores_propios.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&valores_propios](int i1, int i2) {
        return valores_propios(i1) > valores_propios(i2);
        });

    Eigen::VectorXd valores_propios_ordenados(valores_propios.size());
    Eigen::MatrixXd vectores_propios_ordenados(matriz_correlaciones.cols(), matriz_correlaciones.cols());
    for (int i = 0; i < indices.size(); ++i) {
        valores_propios_ordenados(i) = valores_propios(indices[i]);
        vectores_propios_ordenados.col(i) = vectores_propios.col(indices[i]);
    }

    // 4. Construir la matriz V
    Eigen::MatrixXd V = vectores_propios_ordenados;

    // matriz de componentes principales (C)
    Eigen::MatrixXd C = matriz_estandarizada * V;

    // - Matriz de calidades de los individuos (Q)
    Eigen::MatrixXd Q = C.array().square();

    // - Matriz de coordenadas de variables (T)
    Eigen::MatrixXd T = V.transpose();

    // - Vector de inercias de los ejes (I)
    Eigen::VectorXd I = valores_propios_ordenados.transpose();

    // - Matriz de calidades de variables (S)
    Eigen::MatrixXd S = vectores_propios_ordenados.array().square();

    // Mostrar resultados
    cout << "Matriz de componentes principales (C):\n" << C << endl;
    cout << "Matriz de calidades de los individuos (Q):\n" << Q << endl;
    cout << "Matriz de coordenadas de variables (T):\n" << T << endl;
    cout << "Vector de inercias de los ejes (I):\n" << I << endl;
    cout << "Matriz de calidades de variables (S):\n" << S << endl;




    Eigen::MatrixXd datos(4, 3);
    datos << 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0;

    ACP acp(datos);
    acp.calcularACP();
    acp.calcularMatrizCalidadIndividuos();
    acp.calcularMatrizCoordenadasVariables();
    acp.calcularMatrizCalidadVariables();
    acp.calcularVectorInercias();
    acp.mostrarResultados();

    return 0;
}
