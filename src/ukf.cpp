#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    is_initialized_ = false;

    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial time stamp
    time_us_ = 0;

    // state dimension
    n_x_ = 5;

    // augmented state dimension
    n_aug_ = 7;

    // initial state vector
    x_ = VectorXd(n_x_);
    x_ << 0,0,0,0,0;

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    P_ <<   0.0043,  -0.0013, 0.0030, -0.0022, -0.0020,
            -0.0013, 0.0077,  0.0011, 0.0071,  0.0060,
            0.0030,  0.0011,  0.0054, 0.0007,  0.0008,
            -0.0022, 0.0071,  0.0007, 0.0098,  0.0100,
            -0.0020, 0.0060,  0.0008, 0.0100,  0.0123;

    // initial predicted sigma points
    Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

    // lambda
    lambda_ = 3 - n_x_;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 30;

    // initial weights
    weights_ = VectorXd(5);

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    // initial lidar NIS
    NIS_laser_ = 0.0;

    // initial radar NIS
    NIS_radar_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    while (!is_initialized_){
        cout << "searching for non-zero measurement... " << endl;

        if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) &&
            (meas_package.raw_measurements_[0]* meas_package.raw_measurements_[1] != 0)) {
            time_us_ = meas_package.timestamp_;
            cout << "initial time: " << time_us_ << endl;

            double rho = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];
            double rhod = meas_package.raw_measurements_[2];

            x_ << cos(phi) * rho, sin(phi) * rho, 0, 0, 0;
            cout << "initial x (1st non-zero radar measurement):" << endl;
            cout << x_ << endl;
            cout << "-----------------------------" << endl;
            is_initialized_ = true;

        }
        else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) &&
                 (meas_package.raw_measurements_[0]* meas_package.raw_measurements_[1] != 0)){
            time_us_ = meas_package.timestamp_;
            cout << "initial time: " << time_us_ << endl;

            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
            cout << "initial x (1st non-zero laser measurement):" << endl;
            cout << x_ << endl;
            cout << "-----------------------------" << endl;
            is_initialized_ = true;
        }
        return;
    }
    double dt = (double)(meas_package.timestamp_-time_us_)/1000000.0;
    cout << "delta t: " << dt << endl;
    time_us_ = meas_package.timestamp_;
    Prediction(dt);


}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

    /**
     * generate augmented sigma points
    */
    // create augmented mean vector (7)
    VectorXd x_aug = VectorXd(n_aug_);
    //create augmented state covariance (7*7)
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    //create sigma point matrix (7*15)
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_*std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i< n_aug_; i++)
    {
        Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }
    cout << "Xsig_aug:" << endl;
    cout << Xsig_aug << endl;
    /**
     * predict sigma points
    */


}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */
}
