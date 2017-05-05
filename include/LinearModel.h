#ifndef LINEARMODEL_H
#define LINEARMODEL_H
#include <vector>
#include <map>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

class LinearModel
{
    public:
        LinearModel(VectorXd w, double b);
        LinearModel random_init(int num_inputs, float std);
        VectorXd weights;
        map<char, VectorXd> compute_activations(MatrixXd X);
        double biases;
        VectorXd get_predictions(map<char, VectorXd> act);
        map<char, VectorXd> cost_deriv(MatrixXd X, map<char, VectorXd> act, VectorXd dldz);
        void grad_desc_update(map<char, VectorXd> derivs, double learning_rate);
        void decay_weights(double decay_param, double learning_rate);

    protected:

    private:

};

class LogisticSquaredError

{
    public:
        VectorXd value(VectorXd z, VectorXd t);
        VectorXd deriv(VectorXd z, VectorXd t);

};

void train_model(LinearModel model, MatrixXd X, VectorXd t, int num_epochs, double learning_rate, int print_every, double weight_decay);


#endif // LINEARMODEL_H
