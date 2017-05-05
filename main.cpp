#include <vector>
#include "LinearModel.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <stdlib.h>     /* srand, rand */

using namespace std;
using namespace Eigen;

VectorXd random_vec_init(int num_samples){
    VectorXd ret(num_samples);

    for(int i = 0; i< num_samples; i++){
        int v1 = rand() %100;
        if(v1 % 2 == 0){
           ret[i] = 0;
        }
        else{
            ret[i] = 1;
        }
    }
    return ret;
}
int main() {
    int num_samples = 600;
    VectorXd v(4);
    MatrixXd m;
//    LinearModel temp;
    v[0] = 10.0;
    LinearModel Lin1(v,1);

    Lin1 = Lin1.random_init(2,1);
    LogisticSquaredError lse;

    VectorXd log = lse.value(v,v);

    VectorXd v1= random_vec_init(num_samples);
    VectorXd v2= random_vec_init(num_samples);

    MatrixXd X(num_samples, 2);
    X.col(0) = v1;
    X.col(1) = v2;

    VectorXd t = v1.cwiseProduct(v2);
    train_model(Lin1, X, t, 1000, 0.1, 10, 0.001);
    return 0;
}
