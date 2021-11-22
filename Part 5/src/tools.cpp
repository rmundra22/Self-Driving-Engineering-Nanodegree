#include "tools.h"
#include <iostream>
#include<cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
	VectorXd rmse_values(4);
	rmse_values << 0, 0, 0, 0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if(estimations.size() != ground_truth.size() || estimations.size() == 0){
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	for (int i = 0; i < estimations.size(); i++) { 
        for (int j = 0; j < estimations[i].size(); j++){
        	rmse_values[j] += pow((estimations[i][j] - ground_truth[i][j]), 2)
        }
        rmse_values[i] = sqrt(rmse_values[i])
    }

    M_RMSE.push_back(rmse_values);
	return M_RMSE
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
}
