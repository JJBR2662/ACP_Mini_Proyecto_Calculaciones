#include <iostream>
#include <Eigen/Dense>
#include "ACP.h"

int main() {
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
