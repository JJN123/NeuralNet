#include "LinearModel.h"
#include <map>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>

using namespace std;
using namespace Eigen;

LinearModel::LinearModel(VectorXd w, double b):
weights(w), biases(b)
{
   // cout << "weights size cons "<< weights.size() << endl;;
   // weights = w;
    //biases = b;
}

LinearModel LinearModel::random_init(int num_inputs, float std)
{
   //w = random_vec(num_inputs)  /// TOdO Randomize
   float b;
   VectorXd w = VectorXd::Random(num_inputs);
   b = w[0];
   LinearModel LM(w, b);
   return LM;
}
map<char, VectorXd> LinearModel::compute_activations(MatrixXd X)
{ // using map for activations, which will be useful for models with more than one layer.
    VectorXd bias_vec = VectorXd::Constant(X.rows(), biases);
    //cout << bias_vec.size() << bias_vec.cols() << endl;
    VectorXd z = X*weights + bias_vec;
    //cout << weights.cols() << endl;
    map<char, VectorXd> ret;
    ret['z'] = z;
    return ret;

}

VectorXd LinearModel::get_predictions(map<char, VectorXd> act)
{
    VectorXd preds(act['z'].size());
    //cout << act['z'].size() << endl;
    for(int i = 0; i< act['z'].size(); i++){
        if(act['z'][i] > 0){
            preds[i] = 1;
    }
    else{
        preds[i] = 0;
    }
    return preds;
}
}

map<char, VectorXd> LinearModel::cost_deriv(MatrixXd X, map<char, VectorXd> act, VectorXd dldz)
{
    //remember to only use el from derivs['b']
    map<char, VectorXd> derivs;
   // cout << "dldz" << dldz.size() << endl;
    derivs['w'] = X.transpose()*dldz / X.rows();

    derivs['b'] = VectorXd::Constant(weights.size(), dldz.mean());

    return derivs;
}

void LinearModel::grad_desc_update(map<char, VectorXd> derivs, double learning_rate){
    weights -= learning_rate*derivs['w'];
    biases -= learning_rate*derivs['b'][0];

}

void LinearModel::decay_weights(double decay_param, double learning_rate)
{
    weights *= 1-decay_param*learning_rate;
}



void train_model(LinearModel model, MatrixXd X, VectorXd t, int num_epochs = 1000, double learning_rate = 0.1, int print_every = 10, double weight_decay=0.001)
//Pass in initialized model, model param updates persist across epochs
{

                for(int i = 0; i < num_epochs; i++){

                    map<char, VectorXd> act = model.compute_activations(X);
                    LogisticSquaredError ls;
                    VectorXd dldz = ls.deriv(act['z'], t);
                    map<char, VectorXd> param_derivs = model.cost_deriv(X, act, dldz);
                    model.grad_desc_update(param_derivs, learning_rate);
                    model.decay_weights(weight_decay ,learning_rate);
                    if(i % print_every == 0){
                        VectorXd preds = model.get_predictions(act);
                        VectorXd train_loss = ls.value(act['z'], t);

                        //cout << "Epoch" << i << "train loss is:" << train_loss << endl;
                        //cout  << "Prediction is" << preds[0] << endl;

                    }
                }
                map<char, VectorXd> act = model.compute_activations(X);
                VectorXd preds = model.get_predictions(act);
                cout << act['z'] << endl;
                MatrixXd final(preds.size(), 3);
                final.col(0) = X.col(0);
                final.col(1) = X.col(1);
                final.col(2) = act['z'];
                //final.col(3) = preds;
                std::string sep = "\n----------------------------------------\n";
                IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
                std::cout << "Input is" << final.format(CleanFmt) << sep;// "Prediction is" << preds << endl;
}

VectorXd LogisticSquaredError::value(VectorXd z, VectorXd t)
{
    VectorXd y = 1/(1+exp(-1*z.array()));

    return (0.5)*(y-t).cwiseProduct(y-t);

}
VectorXd LogisticSquaredError::deriv(VectorXd z, VectorXd t)
{
    //VectorXd y;
    VectorXd y = 1/(1+exp(-1*z.array()));
    //cout << "y size" << y.size() << "t size "<< t.size() << endl;
    //return y-t;
    y = (y-t).cwiseProduct(y);//*(VectorXd::Constant(y.size(), 1)-y);
    return y.cwiseProduct(VectorXd::Constant(y.size(), 1)-y);
}
