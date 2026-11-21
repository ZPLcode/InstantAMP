// Floating-base inverse dynamics for a delayed Zenbot with two fixed sole anchors.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace {
constexpr int N=29,F=32,J=23;
using Vec=std::array<double,N>;
void orientation(const double* mat,const double* target,double* err){
  double quat[4],inv[4],dq[4];mju_mat2Quat(quat,mat);mju_negQuat(inv,quat);mju_mulQuat(dq,target,inv);
  mju_quat2Vel(err,dq,1.);
}
void nonnegative(const double a[F][N],const double* effort,double prior,double* result){
  double normal[F][F]={},rhs[F]={};
  for(int k=0;k<F;++k){
    for(int j=0;j<F;++j)for(int r=0;r<6;++r)normal[k][j]+=a[k][r]*a[j][r]*(r<3?1.:36.);
    for(int r=0;r<6;++r)rhs[k]+=a[k][r]*effort[r]*(r<3?1.:36.);
    normal[k][k]+=.025*.025;rhs[k]+=.025*.025*prior;
  }
  bool active[F]={};double z[F]={};std::fill_n(result,F,0.);
  for(int outer=0;outer<3*F;++outer){
    int next=-1;double best=1.e-8;
    for(int k=0;k<F;++k)if(!active[k]){
      double g=rhs[k];for(int j=0;j<F;++j)g-=normal[k][j]*result[j];
      if(g>best){best=g;next=k;}
    }
    if(next<0)break;
    active[next]=true;
    for(int inner=0;inner<F;++inner){
      int ids[F],n=0;for(int k=0;k<F;++k)if(active[k])ids[n++]=k;
      if(!n)break;
      double H[F*F],r[F],sol[F];
      for(int k=0;k<n;++k){r[k]=rhs[ids[k]];for(int j=0;j<n;++j)H[k*n+j]=normal[ids[k]][ids[j]];}
      mju_cholFactor(H,n,1.e-12);mju_cholSolve(sol,H,r,n);
      std::fill_n(z,F,0.);bool valid=true;double alpha=1.;
      for(int k=0;k<n;++k){int index=ids[k];z[index]=sol[k];if(sol[k]<=0.){
        valid=false;double v=result[index]/(result[index]-sol[k]+1.e-30);alpha=std::min(alpha,v);
      }}
      if(valid){std::copy_n(z,F,result);break;}
      for(int k=0;k<F;++k)if(active[k]){result[k]+=alpha*(z[k]-result[k]);if(result[k]<=1.e-9){result[k]=0.;active[k]=false;}}
    }
  }
}
}

extern "C" void stance(const mjModel* m,mjData* d,const int* bodies,const double* motor,const double* neutral,
                       const double* anchors,double look,double gain,double* result) {
  mj_forward(m,d);
  double H[N*N]={},rhs[N]={},contact[F][N]={};
  for(int i=0;i<N;++i)H[i*N+i]=.001;
  auto task=[&](const double* j,int rows,const double* e,double rate,double w,const double* deriv=nullptr){
    for(int k=0;k<rows;++k){
      const double* a=j+k*N;double vel=0.,bias=0.;
      for(int l=0;l<N;++l){vel+=a[l]*d->qvel[l];if(deriv)bias+=deriv[k*N+l]*d->qvel[l];}
      double y=std::clamp(rate*rate*e[k]-2.*rate*vel,-80.,80.)-bias;
      for(int l=0;l<N;++l)if(a[l]!=0.){
        rhs[l]+=w*w*a[l]*y;for(int s=0;s<N;++s)H[l*N+s]+=w*w*a[l]*a[s];
      }
    }
  };
  auto at=[&](int b,double x,double y,double z,double* p){
    const double* mat=d->xmat+9*b;for(int j=0;j<3;++j)p[j]=d->xpos[3*b+j]+mat[3*j]*x+mat[3*j+1]*y+mat[3*j+2]*z;
  };
  double jp[3*N],jr[3*N],jd[3*N],rd[3*N],err[3],target[4];
  for(int f=0;f<2;++f){
    int b=bodies[f];double p[3];at(b,.045,0.,-.033,p);const double *rot=d->xmat+9*b;
    const double h=.5*std::atan2(rot[3],rot[0]);target[0]=std::cos(h);target[1]=target[2]=0.;target[3]=std::sin(h);
    mj_jac(m,d,jp,jr,p,b);mj_jacDot(m,d,jd,rd,p,b);
    err[0]=std::clamp(anchors[2*f]-p[0],-.09,.09);err[1]=std::clamp(anchors[2*f+1]-p[1],-.09,.09);err[2]=.003-p[2];
    task(jp,3,err,7.5*gain,5.,jd);orientation(rot,target,err);task(jr,3,err,6.8*gain,3.5,rd);
    int c=f*16;
    for(double x:{-.088,.088})for(double y:{-.031,.031}){
      at(b,.045+x,y,-.033,p);mj_jac(m,d,jp,nullptr,p,b);
      for(int r=0;r<4;++r){
        int axis=r/2;double friction=(r%2?-.7:.7);
        for(int k=0;k<N;++k)contact[c][k]=jp[2*N+k]+friction*jp[N*axis+k];
        c++;
      }
    }
  }
  mj_jacSubtreeCom(m,d,jp,1);
  for(int k=0;k<3;++k)err[k]=(k<2?.5*(anchors[k]+anchors[2+k]):.715)-d->subtree_com[3+k];
  task(jp,3,err,3.9*gain,5.);
  int torso=bodies[2];const double* rot=d->xmat+9*torso;
  double yaw=.5*std::atan2(rot[3],rot[0]);double a[4]={std::cos(.0125),0.,std::sin(.0125),0.},b[4]={std::cos(yaw),0.,0.,std::sin(yaw)};
  mju_mulQuat(target,a,b);orientation(rot,target,err);mj_jac(m,d,nullptr,jr,d->xpos+3*torso,torso);task(jr,3,err,6.*gain,2.4);
  for(int j=0;j<J;++j){
    int k=6+j;H[k*N+k]+=.09;
    rhs[k]+=.09*std::clamp(3.7*3.7*(neutral[j]-d->qpos[7+j])-7.4*d->qvel[k],-80.,80.);
  }
  double accel[N],M[N*N],effort[N],weight[F];
  mju_cholFactor(H,N,1.e-12);mju_cholSolve(accel,H,rhs,N);mj_fullM(m,d,M);
  for(int k=0;k<N;++k){effort[k]=d->qfrc_bias[k]-d->qfrc_passive[k];for(int j=0;j<N;++j)effort[k]+=M[k*N+j]*accel[j];}
  nonnegative(contact,effort,m->body_subtreemass[1]*9.81/32.,weight);
  for(int j=0;j<J;++j){
    double tau=effort[6+j];for(int s=0;s<F;++s)tau-=weight[s]*contact[s][6+j];
    tau=std::clamp(tau,-motor[2*J+j],motor[2*J+j]);
    double target=d->qpos[7+j]+look*d->qvel[6+j]+(tau+motor[J+j]*(d->qvel[6+j]+look*accel[6+j]))/motor[j];
    result[j]=4.*(target-neutral[j]);
  }
}
