//
//  main.cpp
//  BIOS615_Final_project_YJHeo
//
//  Created by Youngjin Heo on 11/19/16.
//  Copyright © 2016 Youngjin Heo. All rights reserved.
//
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <ctime>
#include <cmath>
#include <boost/random/uniform_real.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/mersenne_twister.hpp>

using namespace std;
using namespace Eigen;
using namespace boost;

//MatrixXd betaUpdate(const Ref<const MatrixXd>& XtX,const Ref<const VectorXd>& gamma, double sigma_sq,  const Ref<const MatrixXd>& Y, int p);
MatrixXd betaUpdate(const Ref<const MatrixXd>& X, const Ref<const VectorXd>& gamma, double sigma_sq, const Ref<const MatrixXd>& Y, int p, int K, int n, int B, MatrixXd &beta);
//MatrixXd phiUpdate(const Ref<const MatrixXd>& beta, const Ref<const MatrixXd>& tau_2, double w, double nu0);
MatrixXd phiUpdate(const Ref<const MatrixXd>& beta, const Ref<const MatrixXd>& tau_2, double w, double nu0, int K);
//MatrixXd tauUpdate(const Ref<const MatrixXd>& beta, const Ref<const MatrixXd>& phi, double a1, double a2);
MatrixXd tauUpdate(const Ref<const MatrixXd>& beta, const Ref<const MatrixXd>& phi, double a1, double a2, int K);
double wUpdate(VectorXd& gammaVec);
double sigmaUpdate(const Ref<const MatrixXd>& YVec, const Ref<const MatrixXd>& betaVec, double b1, double b2, const Ref<const MatrixXd>& X);
VectorXd gammaUpdate(const Ref<const MatrixXd>& phi, const Ref<const MatrixXd>& tau_2);
double beta_k_var(MatrixXd &beta_k, int K2);
void validation(MatrixXd &beta_draw, int K3, int tmp);

int main(){
    /*
     input variable :
     user-specified inputs :
     - all matrices and vectors are double type.
     X (matrix): covariates data (n-by-K)
     Y (matrix): response data (n-by-1)
     p (int): the size of blocks (now, it's only available for divisor of K; ex. if K = 12, then p can be 2, 3, 4, 6)
     sigma_sq (double) : sigma^2
     gamma (vector): gamma vector
     
     set-up :
     n : the number of observation (row # of X)
     K : the number of covariates (col # of X)
     B : the number of blocks i.e. B = K/p
     
     at ith iteration:
     
     */
    int n = 8;
    int K = 14;  // the number of covariates
    int p = 3;  // the size of a block
    int B = floor(K/p);  // the number of blocks i.e. K = B*p
    double a1 = 3.0, a2 = 1.0, b1=1.0, b2=1.0;
    double nu0 = 0.0002, sigma_sq = 3.2, w=0.5;
    //    MatrixXd X(n,K);  // input X
    //    MatrixXd Y(n,1);  // input Y*
    MatrixXd Y = MatrixXd::Random(n,1);  // Y* -> temporarily generated with random number for test
    MatrixXd X = MatrixXd::Random(n,K);  // X -> temporarily generated with random number for test
    
    MatrixXd beta = MatrixXd::Random(K,1);  // initialize beta
    
    VectorXd gamma = VectorXd::Random(K);  // temporarily generated with random number for test
    gamma = gamma.cwiseAbs();
    MatrixXd tau_2=MatrixXd::Random(K,1);
    MatrixXd phi=MatrixXd::Random(K,1);
    int iteNum=100000; int keep=50; int thin = 50; int temp=0;  // temp = burning, thin = thinning
    if((iteNum-(iteNum/keep))/thin==(int)(iteNum-(iteNum/keep))/thin){
        temp = iteNum/2;
    }else{
        temp = (int)(iteNum/2) - 1;
    }
    cout << "temp= " << temp << endl;
    int tmp = temp/thin;
    cout << "tmp= " << tmp << endl;
    
    MatrixXd beta_draw(K,tmp);
    for(int i=0; i < iteNum; i++){
        //cout << "i = "<< i << endl;
        beta=betaUpdate(X, gamma, sigma_sq, Y, p, K, n, B, beta);
        phi=phiUpdate(beta, tau_2, w, nu0, K);
        tau_2=tauUpdate(beta, phi, a1, a2, K);
        w=wUpdate(gamma);
        sigma_sq=sigmaUpdate(Y, beta, b1, b2, X);
        gamma=gammaUpdate(phi, tau_2);
        if(i > temp-1 && (double) (i % keep) == 0){
            cout << "i = "<< i << endl;
            for(int kk=0; kk<K; kk++){
                beta_draw(kk,(int)(i-temp)/thin) = beta(kk,0);
            }
            //cout << beta << endl;
            //cout << beta_draw << endl;
        }
    }
    
    
//    validation(beta_draw, K, tmp);
    // validation
    MatrixXd M(K,1); // mean
    M = beta_draw.rowwise().mean();  // mean(beta_k) for all iteration for each k = 1,..,K
    
    MatrixXd e(K,1); // error
    e = M - beta_draw.block(0,0,K,1);
    
    MatrixXd sd(K,1); // sd
    
    cout << "error = \n" << e << endl;
    
    MatrixXd beta_k(1,tmp);
    
    // Final beta
    MatrixXd beta_sol(K,1);
    beta_sol = beta_draw.block(0,tmp-1,K,1);
    
    cout << "beta_sol = \n" << beta_sol << endl;
    
    for (int k = 0 ; k < K ; k++){
        beta_k = beta_draw.block(k,0,1,tmp);
        sd(k,0) = sqrt(beta_k_var(beta_k, K));
        
        boost::math::normal dist(M(k,0), sd(k,0));
        double qLB = quantile(dist, 0.025);
        double qUB = quantile(dist, 0.975);

        cout << qLB << " , " << beta_sol(k,0) << " , " << qUB << endl;
        
        if ((qLB < beta_sol(k,0)) && (beta_sol(k,0) < qUB)){
            cout << "k = " << k << ", True" << endl;
        }else{
            cout << "k = " << k << ", False" << endl;
        }

    }
    

}

//void validation(MatrixXd &beta_draw, int K3, int tmp){
//    MatrixXd M(K3,1); // mean
//    M = beta_draw.rowwise().mean();  // mean(beta_k) for all iteration for each k = 1,..,K
//    
//    MatrixXd e(K3,1); // error
//    e = M - beta_draw.block(0,0,K3,1);
//    
//    MatrixXd sd(K3,1); // sd
//    
////    cout << "error = \n" << e << endl;
//    
//    MatrixXd beta_k(1,tmp);
//    
//    // Final beta
//    MatrixXd beta_sol(K3,1);
//    beta_sol = beta_draw.block(0,tmp-1,K3,1);
//    
//    cout << "beta_sol = \n" << beta_sol << endl;
//    
//    for (int k = 0 ; k < K3 ; k++){
//        beta_k = beta_draw.block(k,0,1,tmp);
//        sd(k,0) = sqrt(beta_k_var(beta_k, K3));
//        
//        boost::math::normal dist(M(k,0), sd(k,0));
//        double qLB = quantile(dist, 0.025);
//        double qUB = quantile(dist, 0.975);
//        
//        cout << qLB << " < " << M(k,0) << " < " << qUB << endl;
//        
//        if ((qLB < beta_sol(k,0)) && (beta_sol(k,0) < qUB)){
//            cout << "k = " << k << ", True" << endl;
//        }else{
//            cout << "k = " << k << ", False" << endl;
//        }
//        
//    }
//    
//    //    boost::math::normal dist(0.0, 1.0);
//    //    // quantile
//    //    double qUB = quantile(dist, 0.975);
//    //    double qLB = quantile(dist, 0.025);
//}

double beta_k_var(MatrixXd &beta_k, int K2){
    double xbar = 0.0, x = 0.0, S2 = 0.0;
    
    for (int i = 0 ; i < K2; i++){
        x = beta_k(0,i);
        xbar += x;
    }
    xbar /= K2;
    for (int i = 0 ; i < K2; i++){
        x = beta_k(0,i);
        double xtilde = x - xbar;
        xtilde *= xtilde;
        S2 += xtilde;
    }
    if(K2>2)
        S2 = S2/(1.0*(K2-1));

    return S2;
}



//MatrixXd betaUpdate(const Ref<const MatrixXd>& X,const Ref<const VectorXd>& gamma, double sigma_sq,  const Ref<const MatrixXd>& Y, int p){
MatrixXd betaUpdate(const Ref<const MatrixXd>& X, const Ref<const VectorXd>& gamma, double sigma_sq, const Ref<const MatrixXd>& Y, int p, int K, int n, int B, MatrixXd &beta){
//    int n = Y.rows();
//    int K = gamma.size();  // the number of covariates
//    int B = floor(K/p);

    MatrixXd Gamma=gamma.asDiagonal();  // Gamma diagonal matrix
    MatrixXd Gamma_j_inv(p,p);  // Gamma_(j)_inverse
    MatrixXd XtX(K, K);  // X'*X
    XtX = X.transpose()*X;
    MatrixXd XtX_j1(p,p); // X'(j)X(j)
    MatrixXd X_wo_j(n, K-p);  // X(-j)
//    MatrixXd beta = MatrixXd::Random(K,1);  // beta from the previous iteration -> temporarily generated with random number for test
    MatrixXd beta_j(p,1);  // beta_(j)
    MatrixXd beta_wo_j(K-p,1);  // beta_(-j)
    MatrixXd Sigma_j_inv(p, p);  // Sigma_(j)_inverse
    MatrixXd mu_j(p, 1);  // mu_(j)
    VectorXd N(p);  // allocation for generating random number
    random_device rd;
    std::mt19937 gen(rd());
    normal_distribution<> normal(0,1);
    double rnorm;

    for (int j = 0 ; j < B ; j++){
        
        // 1. X'(j)*X(j)
        XtX_j1 = XtX.block(p*j,p*j,p,p);
        
        // 2. Gamma_(j)_inverse
        Gamma_j_inv = (Gamma.block(p*j,p*j,p,p)).inverse();
        
        // 3. Sigma_(j)_inverse
        Sigma_j_inv = (XtX_j1 + sigma_sq*n*Gamma_j_inv);
        
        // 4. cholesky decomosition of Sigma_(j)_inverse
        LLT<MatrixXd> chol(Sigma_j_inv);
        MatrixXd L = chol.matrixL();
        
        // 5. beta_(-j) : combine two slices before and after beta(j)
        //    X_(-j) : combine two slices before and after X(j)
        
        if (j == 0){
            beta_wo_j = beta.block(p,0, K-p,1);
            X_wo_j = X.block(0,p,n,K-p);
        }else{
            if(((K % p) == 0) && (j == B-1)){
                beta_wo_j = beta.block(0, 0, K-p,1);
                X_wo_j = X.block(0,0,n,K-p);
            }else{
                beta_wo_j.block(0,0,p*j,1) = beta.block(0,0,p*j,1);
                beta_wo_j.block(p*j,0,K-p*(j+1),1) = beta.block(p*(j+1),0, K-p*(j+1),1);
                
                X_wo_j.block(0,0,n,p*j) = X.block(0,0,n,p*j);
                X_wo_j.block(0,p*j,n,K-p*(j+1)) = X.block(0,p*(j+1),n,K-p*(j+1));
            }
        }
        
        
        // 6. mu_(j) : solve Sigma_(j)_inv * mu_(j) = X(j)'*(Y - X(-j)*beta(-j))
        mu_j = Sigma_j_inv.jacobiSvd(ComputeThinU | ComputeThinV).solve(((X.block(0,p*j, n, p)).transpose())*(Y -X_wo_j*beta_wo_j));
        
        // 7. draw a multivariate normal sample beta(j) from MVN(mu_j, sigma_sq*Sigma_j) where Sigma_j = cholestky decomposed by L*t(L)
        // 7.1. generate Vector N (u1,...,up) where ui ~ N(0,1)
        for (int n = 0; n < p ; n++){
            rnorm = normal(gen);
            N(n) = rnorm;
        }
        
        // 7.2. Generate beta from MVN(mu_j, Sigma_j)
        beta_j = mu_j + (1/sigma_sq)*(L.inverse())*N;
        
        beta.block(p*j,0,p,1) = beta_j;
        
    }

    // Last non-multiple block
    if ((K % p) != 0){
        int J = p*B;
        int rem_p = K-p*B;
        
        // 1. X'(j)*X(j)
        XtX_j1.resize(rem_p,rem_p);
        XtX_j1 = XtX.block(J,J,rem_p,rem_p);
        
        // 2. Gamma_(j)_inverse
        Gamma_j_inv.resize(rem_p,rem_p);
        Gamma_j_inv = (Gamma.block(J,J,rem_p,rem_p)).inverse();
        
        // 3. Sigma_(j)_inverse
        Sigma_j_inv.resize(rem_p,rem_p);
        Sigma_j_inv = (XtX_j1 + sigma_sq*n*Gamma_j_inv);
        
        // 4. cholesky decomosition of Sigma_(j)_inverse
        LLT<MatrixXd> chol(Sigma_j_inv);
        MatrixXd L = chol.matrixL();
        
        // 5. beta_(-j) : combine two slices before and after beta(j)
        //    X_(-j) : combine two slices before and after X(j)
        
        int K1 = K-rem_p;
        beta_wo_j.resize(K1,1);
        beta_wo_j = beta.block(0, 0, K1,1);
        
        X_wo_j.resize(n, K1);
        X_wo_j = X.block(0,0,n,K1);
        
        // 6. mu_(j) : solve Sigma_(j)_inv * mu_(j) = X(j)'*(Y - X(-j)*beta(-j))
        mu_j = Sigma_j_inv.jacobiSvd(ComputeThinU | ComputeThinV).solve(((X.block(0,J, n, rem_p)).transpose())*(Y -X_wo_j*beta_wo_j));
        
        // 7. draw a multivariate normal sample beta(j) from MVN(mu_j, sigma_sq*Sigma_j) where Sigma_j = cholestky decomposed by L*t(L)
        // 7.1. generate Vector N (u1,...,urep_p) where ui ~ N(0,1)
        N.resize(rem_p);
        for (int n = 0; n < rem_p ; n++){
            rnorm = normal(gen);
            N(n) = rnorm;
        }
        
        // 7.2. Generate beta from MVN(mu_j, Sigma_j)
        beta_j.resize(rem_p,1);
        beta_j = mu_j + (1/sigma_sq)*(L.inverse())*N;
        
        beta.block(J,0,rem_p,1) = beta_j;
        
    }
    
    
    return beta;
}

//MatrixXd phiUpdate(const Ref<const MatrixXd>& beta, const Ref<const MatrixXd>& tau_2, double w, double nu0){
MatrixXd phiUpdate(const Ref<const MatrixXd>& beta, const Ref<const MatrixXd>& tau_2, double w, double nu0, int K){
    //prior for tau^-2
    uniform_real_distribution<double> uniform(0.0001,1);
    random_device rd;
    std::mt19937 gen(rd());
//    int K = beta.rows();
    double w1, w2;
    MatrixXd phi(K,1);
    for(int i=0; i<K; i++){
        //cout << "w= "<< w << endl;
        //cout << "tau = " << tau_2(i,0) << endl;
        //cout << "beta = " << beta(i,0) << endl;
        //cout << "Inside of exp func " << pow(beta(i,0)/tau_2(i,0),2)/(2*nu0) << endl;
        w1 = (1-w)*pow(nu0, -1/2)*exp(-pow(beta(i,0)/tau_2(i,0),2)/(2*nu0));
        w2 = w*exp(-pow(beta(i,0)/tau_2(i,0),2)/2);
        double target = w1/(w1+w2)*(nu0-1)+1;
        //cout << "w1 = " << w1 << endl;
        //cout << "w2 = " << w2 << endl;
        //cout << "target = " << target << endl;
        if(uniform(gen)<target){
            phi(i,0) = nu0; // phi = nu0
        }else{
            phi(i,0) = 1;
        }
        //cout << "phi = " << phi(i,0) << endl;
    }
    return phi;
}

//MatrixXd tauUpdate(const Ref<const MatrixXd>& beta, const Ref<const MatrixXd>& phi, double a1, double a2){
MatrixXd tauUpdate(const Ref<const MatrixXd>& beta, const Ref<const MatrixXd>& phi, double a1, double a2, int K){
    uniform_real_distribution<double> uniform(0.0001,1);
    random_device rd;
    std::mt19937 gen(rd());
    //prior for tau^-2
//    int K = beta.rows();
    MatrixXd tau_2(K,1);
    for(int i=0; i<K; i++){
        gamma_distribution<> gamma(a1*1.0+(1/2), a2*1.0+pow(beta(i,0), 2)/(2*phi(i,0)));
        tau_2(i,0) = 1/gamma(gen);
    }
    return tau_2;
}

double wUpdate(VectorXd& gammaVec){
    std::mt19937 rng;
    rng.seed(time(0));
    int num1=0, num0=0;
    for(int i=0; i < gammaVec.size(); i++){
        if(gammaVec(i)==1) num1++;
        else num0++;
    }
    //cout << "# of 0: " << num0 << " and # of 1: " << num1 <<endl;
    uniform_real<> uni(0,1);
    math::beta_distribution<> beta(1+num1,1+num0);
    double y=uni(rng);
    double randFromDist = quantile(beta, y);
    return randFromDist;
}

double sigmaUpdate(const Ref<const MatrixXd>& YVec, const Ref<const MatrixXd>& betaVec, double b1, double b2, const Ref<const MatrixXd>& X){
    std::mt19937 rng;
    rng.seed(time(0));
    uniform_real<> uni(0,1);
    double y=uni(rng);
    int n=YVec.rows();
    MatrixXd norms=(YVec-X*betaVec).transpose()*(YVec-X*betaVec);
    //cout << norms(0,0) << endl;
    double a=b1+n/2.0, b=b2+norms(0,0)/(2*n);
    math::gamma_distribution<> gamma(a,b);
    double randFromDist = quantile(gamma, y);
    return randFromDist;
}

VectorXd gammaUpdate(const Ref<const MatrixXd>& phi, const Ref<const MatrixXd>& tau_2){
    int K = phi.rows();
    VectorXd gamma(K);
    for(int i =0; i < K; i++){
        gamma(i,0)=phi(i,0)*tau_2(i,0);//tau_2 is tau^2, not tau^-2
    }
    return gamma;
}
