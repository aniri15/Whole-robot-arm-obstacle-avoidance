#include "../include/panda_ik.hpp"

using namespace std;

int main(){
    array<double, 6> xyzrpy;
    array<double, 7> q_actual;
    array<double, 7> output;

    xyzrpy = {0.4, 0.0, 0.6, 45.0, -45.0, 0.0};
    q_actual = {-0.29, -0.176, -0.232, -0.67, 1.04, 2.56, 0.0};


    output = compute_inverse_kinematics(xyzrpy,q_actual);

    for (int i = 0; i < 7; i++) {
        std::cout << output[i] << " ";
    }
}



