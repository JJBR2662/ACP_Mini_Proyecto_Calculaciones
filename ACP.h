#ifndef ACP_H
#define ACP_H

#include <Eigen/Dense>

class ACP {
private:
    Eigen::MatrixXd X, C, Q, T, S, V;
    Eigen::VectorXd valoresPropios;
    Eigen::RowVectorXd I;

public:
    ACP(const Eigen::MatrixXd& data);
    void calcularACP();
    void calcularMatrizCalidadIndividuos();
    void calcularMatrizCoordenadasVariables();
    void calcularMatrizCalidadVariables();
    void calcularVectorInercias();
    void mostrarResultados();
};

#endif
