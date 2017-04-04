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
    weights_ = VectorXd(2*n_aug_+1);

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
            cout << "-----------------initial state---------------" << endl;
            time_us_ = meas_package.timestamp_;
            cout << "initial time: " << time_us_ << endl;

            double rho = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];
            double rhod = meas_package.raw_measurements_[2];

            x_ << cos(phi) * rho, sin(phi) * rho, 0, 0, 0;
            cout << "initial x (1st non-zero radar measurement):" << endl;
            cout << x_ << endl;
            cout << "-----------------initial state---------------" << endl;
            is_initialized_ = true;

        }
        else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) &&
                 (meas_package.raw_measurements_[0]* meas_package.raw_measurements_[1] != 0)){
            time_us_ = meas_package.timestamp_;
            cout << "-----------------initial state---------------" << endl;
            cout << "initial time: " << time_us_ << endl;

            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
            cout << "initial x (1st non-zero laser measurement):" << endl;
            cout << x_ << endl;
            cout << "-----------------initial state---------------" << endl;
            is_initialized_ = true;
        }
        return;
    }
    double dt = (double)(meas_package.timestamp_-time_us_)/1000000.0;
    cout << "delta t: " << dt << endl;
    time_us_ = meas_package.timestamp_;
    Prediction(dt);
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
        cout << "-----------receive radar measurement---------" << endl;
        UpdateRadar(meas_package);
        cout << "-----------updated x--------------------" << endl;
        cout << x_ << endl;
    }
    if (meas_package.sensor_type_ == MeasurementPackage::LASER){
        cout << "-----------receive laser measurement---------" << endl;
        UpdateLidar(meas_package);
        cout << "-----------updated x--------------------" << endl;
        cout << x_ << endl;
    }


}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    cout << "------------prediction start---------------" << endl;

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
        Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }

    /**
     * predict sigma points
    */
    for (int i = 0; i< 2*n_aug_+1; i++)
    {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        //predicted state values
        double px_pred, py_pred;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_pred = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_pred = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_pred = p_x + v*delta_t*cos(yaw);
            py_pred = p_y + v*delta_t*sin(yaw);
        }

        double v_pred = v;
        double yaw_pred = yaw + yawd*delta_t;
        double yawd_pred = yawd;

        //add noise
        px_pred = px_pred + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_pred = py_pred + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_pred = v_pred + nu_a*delta_t;

        yaw_pred = yaw_pred + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_pred = yawd_pred + nu_yawdd*delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_pred;
        Xsig_pred_(1,i) = py_pred;
        Xsig_pred_(2,i) = v_pred;
        Xsig_pred_(3,i) = yaw_pred;
        Xsig_pred_(4,i) = yawd_pred;
    }

    /**
     * predict mean & covariance
    */
    // set weights
    double weight_0 = lambda_/(lambda_+n_aug_);
    weights_(0) = weight_0;
    for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
        double weight = 0.5/(n_aug_+lambda_);
        weights_(i) = weight;
    }

    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;

    }
    cout << "x_pred:" << endl;
    cout << x_ << endl;
    cout << "p_pred:" << endl;
    cout << P_ << endl;
    cout << "----------------prediction done----------------" << endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    /**
     * Predict measurement
     */
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(2, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);

        // measurement model
        Zsig(0,i) = p_x;
        Zsig(1,i) = p_y;
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(2);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(2,2);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(2,2);
    R <<    std_laspx_*std_laspx_, 0,
            0,                   std_laspy_*std_laspy_;
    S = S + R;

    /**
     * Update State
     */

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, 2);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     * Predict measurement
     */
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
        Zsig(1,i) = atan2(p_y,p_x);                                 //phi
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(3);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(3,3);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(3,3);
    R <<    std_radr_*std_radr_, 0,                       0,
            0,                   std_radphi_*std_radphi_, 0,
            0,                   0,                       std_radrd_*std_radrd_;
    S = S + R;

    /**
     * Update State
     */

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, 3);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

}
