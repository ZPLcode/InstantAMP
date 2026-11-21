// Native delayed-actuator rollouts used to select a physically feasible recovery.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <omp.h>

namespace {
constexpr int J=23, V=29, X=104;
using Motors=std::array<double,J>;
struct Node {
  mjData* d;
  Motors previous, older;
  double cost=0., upright_x=0., upright_y=0.;
  int up=0;
};
void advance(const mjModel* m, mjData* d, const int* lag, const double* motor,
             const Motors& target,const Motors& previous,const Motors& older) {
  for (int sub=0;sub<4;++sub){
    for(int j=0;j<J;++j){
      double u=sub>=lag[j]?target[j]:sub+4>=lag[j]?previous[j]:older[j];
      double v=d->qvel[6+j];
      double tau=std::clamp(motor[j]*(u-d->qpos[7+j])-motor[J+j]*v,-motor[2*J+j],motor[2*J+j]);
      d->ctrl[j]=std::abs(v)>=motor[3*J+j] && tau*v>0.?0.:tau;
    }
    mj_step(m,d);
  }
}
void copy(const mjModel* m,Node& dst,const Node& src){
  mjData* local=dst.d;dst=src;dst.d=local;mj_copyData(local,m,src.d);
}
}

extern "C" void plan(const mjModel* m,const mjData* reset,int n,int beam,int batch,int stride,int threads,int seed,
                     const double* motor,const int* lag,const int* src_lag,const int* feet,const double* q,
                     const double* v,const double* target,const float* gain,const double* neutral,
                     const double* previous,const double* older,const float* actor,double scale,double brake,const double* support,const int* hands,double weight,float* actions,int* parents,
                     double* capture,double* positions){
  int stages=(n+stride-1)/stride, branches=beam*batch;
  std::vector<Node> history(beam), children(branches);
  for(Node &a:history){a.d=mj_makeData(m);mj_copyData(a.d,m,reset);std::copy(previous,previous+J,a.previous.begin());std::copy(older,older+J,a.older.begin());}
  for(Node &a:children)a.d=mj_makeData(m);
  std::vector<float> uc(branches*stride*J); std::vector<int> rank(branches);
  std::iota(rank.begin(),rank.end(),0);
  for(int stage=0;stage<stages;++stage){
    int tick=stage*stride, length=std::min(n-tick,stride);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int branch=0;branch<branches;++branch){
      int parent=branch/batch,variant=branch%batch;
      Node& c=children[branch];copy(m,c,history[parent]);
      std::mt19937 rng(seed+91577*stage+773*branch);std::normal_distribution<double> gauss;
      Motors offset;double sigma=variant<3?0.:variant%4==0?.24:variant%4==1?.14:variant%4==2?.065:.035;
      if(seed>10000)sigma*=2.;
      double amp=variant<3?std::array<double,3>{1.,.65,1.3}[variant]:std::clamp(1.+.35*gauss(rng),.25,1.8);
      double delay_shift=variant<3?.9:std::clamp(.85+.4*gauss(rng),0.,1.5);
      for(int j=0;j<J;++j)offset[j]=sigma*gauss(rng);
      double sum=0.,last=0.;
      for(int i=0;i<length;++i){
        int t=tick+i;double dx[X];mj_differentiatePos(m,dx,1.,q+30*t,c.d->qpos);
        for(int j=0;j<V;++j)dx[V+j]=c.d->qvel[j]-v[V*t+j];
        for(int j=0;j<J;++j){dx[2*V+j]=c.previous[j]-target[J*(t+1)+j];dx[2*V+J+j]=c.older[j]-target[J*t+j];}
        double features[76];
        const double *rot=c.d->qpos+3;
        for(int j=0;j<3;++j)features[j]=.12*c.d->qvel[3+j];
        features[3]=2.*(rot[1]*rot[3]-rot[0]*rot[2]);
        features[4]=2.*(rot[2]*rot[3]+rot[0]*rot[1]);
        features[5]=1.-2.*(rot[1]*rot[1]+rot[2]*rot[2]);
        for(int j=0;j<J;++j){features[6+j]=c.d->qpos[7+j];features[29+j]=.04*c.d->qvel[6+j];features[52+j]=.32*(c.previous[j]-neutral[j]);}
        features[75]=1.;
        Motors cmd;
        for(int j=0;j<J;++j){
          const float* k=gain+(t*J+j)*X;double gc=0.;
          for(int a=0;a<X;++a)gc+=k[a]*dx[a];
          double slope=t+1<n?.5*(target[J*(t+3)+j]-target[J*(t+1)+j]):0.;
          int delta=lag[j]-src_lag[j];
          if(variant%3==1 || seed>10000)
            slope=delta>0?target[J*std::min(t+3,n+1)+j]-target[J*(t+2)+j]:target[J*(t+2)+j]-target[J*(t+1)+j];
          double limit=(variant%3==1 || seed>10000)?2.2:.8;
          double goal=target[J*(t+2)+j]+std::clamp(gc*amp,-.85,.85)+offset[j]+delay_shift*.25*delta*std::clamp(slope,-limit,limit);
          float action=static_cast<float>((goal-neutral[j])*4.);
          double residual=0.;if(scale!=0.)for(int k=0;k<76;++k)residual+=actor[k*J+j]*features[k];
          action+=static_cast<float>(scale*std::tanh(residual));
          uc[(branch*stride+i)*J+j]=action;
          cmd[j]=static_cast<float>(neutral[j])+.25f*action;
        }
        advance(m,c.d,lag,motor,cmd,c.previous,c.older);c.older=c.previous;c.previous=cmd;
        const double *ref=q+30*(t+1),*vel=v+V*(t+1);mj_differentiatePos(m,dx,1.,ref,c.d->qpos);
        double cost=20.*(dx[0]*dx[0]+dx[1]*dx[1])+360.*dx[2]*dx[2];
        for(int j=3;j<6;++j)cost+=36.*dx[j]*dx[j];
        for(int j=6;j<V;++j)cost+=2.3*dx[j]*dx[j];
        for(int j=0;j<V;++j){double dv=c.d->qvel[j]-vel[j];cost+=(j<3?.9:j<6?.18:.008)*dv*dv;}
        double z=c.d->qpos[2], cs=1.-2.*(std::pow(c.d->qpos[4],2)+std::pow(c.d->qpos[5],2));
        if(z>.58 && z<=.78 && cs>.866){if(c.up==0){c.upright_x=c.d->qpos[0];c.upright_y=c.d->qpos[1];} c.up=std::min(10,c.up+1);}
        else if(c.up<10)c.up=0;
        const double* expected=support+2*t;
        if(expected[0]+expected[1]>.08){
          double measured[2]={};
          for(int j=0;j<c.d->ncon;j++){
            const auto& co=c.d->contact[j];int a=m->geom_bodyid[co.geom1],b=m->geom_bodyid[co.geom2];
            if(a&&b)continue;
            for(int side=0;side<2;++side)if(a+b==hands[side]){
              mjtNum force[6];mj_contactForce(m,c.d,j,force);measured[side]+=std::max(0.,force[0])/weight;
            }
          }
          for(int side=0;side<2;++side){double gap=std::max(0.,expected[side]-measured[side]);cost+=65.*gap*gap;}
        }
        if(c.up==10)cost+=12.*(std::pow(c.d->qpos[0]-c.upright_x,2)+std::pow(c.d->qpos[1]-c.upright_y,2));
        double near=std::clamp((z-.43)/.14,0.,1.)*std::clamp((cs-.45)/.4,0.,1.);
        if(near>0.){
          const auto* lf=c.d->xpos+3*feet[0];const auto* rf=c.d->xpos+3*feet[1];
          double com_error=0.,speed=0.;
          for(int j=0;j<2;++j){double e=c.d->subtree_com[3+j]+.245*c.d->qvel[j]-.5*(lf[j]+rf[j]);com_error+=e*e;speed+=std::pow(c.d->qvel[j],2);}
          cost+=near*brake*(70.*com_error+4.*speed+20.*std::pow(lf[2]-rf[2],2));
        }
        sum+=cost;last=cost;
      }
      c.cost=.56*c.cost+sum/length+.5*last;
    }
    std::sort(rank.begin(),rank.end(),[&](int a,int b){return children[a].cost<children[b].cost;});
    std::vector<int> quota(beam),selected;
    for(int r:rank){
      if(stage && quota[r/batch]>=std::max(2,(beam+1)/2))continue;
      if(!stage && std::any_of(selected.begin(),selected.end(),[&](int a){return std::abs(children[a].cost-children[r].cost)<1e-10;}))continue;
      selected.push_back(r);++quota[r/batch];if(static_cast<int>(selected.size())==beam)break;
    }
    while(static_cast<int>(selected.size())<beam)selected.push_back(rank[selected.size()]);
    for(int b=0;b<beam;++b){
      int r=selected[b],off=stage*beam+b;copy(m,history[b],children[r]);Node& c=history[b];
      std::copy_n(&uc[r*stride*J],length*J,actions+off*stride*J);
      parents[off]=r/batch;mj_kinematics(m,c.d);
      const double *lf=c.d->xpos+3*feet[0],*rf=c.d->xpos+3*feet[1],*p=c.d->qpos,*dv=c.d->qvel;
      double cs=1.-2.*(p[4]*p[4]+p[5]*p[5]),z=p[2];
      double cp=std::hypot(p[0]+.22*dv[0]-.5*(lf[0]+rf[0]),p[1]+.22*dv[1]-.5*(lf[1]+rf[1]));
      double score=(z<.425||cs<.61)?1e5:2.7*cp+2.6*std::abs(lf[2]-rf[2])+.18*std::hypot(dv[0],dv[1])+.05*std::abs(dv[2])+.25*std::abs(.62-z)+.18*(1.-cs);
      if(c.up==10)score+=1.3*std::hypot(p[0]-c.upright_x,p[1]-c.upright_y);
      capture[off]=score;
      std::copy_n(c.d->qpos,30,positions+off*30);
    }
  }
  for(auto& a:history)mj_deleteData(a.d);for(auto &a:children)mj_deleteData(a.d);
}

extern "C" void motor_step(const mjModel* m,mjData* d,const int* lag,const double* motor,
                           const double* command,const double* previous,const double* older) {
  Motors u,p,o;std::copy_n(command,J,u.begin());std::copy_n(previous,J,p.begin());std::copy_n(older,J,o.begin());
  advance(m,d,lag,motor,u,p,o);
}

extern "C" void training(const mjModel* m,int batch,int horizon,int threads,const double* motor,const double* neutral,
                         const int* starts,const int* lag,const double* q,const double* v,const double* goals,const float* gains,
                         const int* links,const double* expected_hand,double weight,const float* weights,const float* noise,float sigma,float scale,int seed,
                         float* obs,float* latent,float* reward,float* amp) {
  #pragma omp parallel num_threads(threads)
  {
    mjData* d=mj_makeData(m);
    #pragma omp for schedule(static)
    for(int row=0;row<batch;++row){
      int id=starts[row];std::mt19937 rng(seed+row*2039);std::normal_distribution<double> normal;
      mj_resetData(m,d);std::copy_n(q+30*id,30,d->qpos);std::copy_n(v+V*id,V,d->qvel);
      for(int j=0;j<J;++j)d->qpos[7+j]+=.007*normal(rng);
      d->qvel[0]+=.035*normal(rng);d->qvel[1]+=.035*normal(rng);
      mj_forward(m,d);
      Motors prev,old,cmd;
      std::copy_n(goals+(id-1)*J,J,prev.begin());std::copy_n(goals+(id-2)*J,J,old.begin());
      double anchor[2]={d->qpos[0],d->qpos[1]};bool rose=false;
      for(int t=0;t<horizon;++t){
        int k=t+id, ix=(row*horizon+t);
        double dx[X];mj_differentiatePos(m,dx,1.,q+30*k,d->qpos);
        for(int j=0;j<V;++j)dx[V+j]=d->qvel[j]-v[V*k+j];
        for(int j=0;j<J;++j){dx[2*V+j]=prev[j]-goals[J*(k-1)+j];dx[2*V+J+j]=old[j]-goals[J*(k-2)+j];}
        float* f=obs+76*ix;double* rot=d->qpos+3;
        for(int j=0;j<3;++j)f[j]=.12*d->qvel[3+j];
        f[3]=2*(rot[1]*rot[3]-rot[0]*rot[2]);f[4]=2*(rot[2]*rot[3]+rot[0]*rot[1]);f[5]=1.-2*(rot[1]*rot[1]+rot[2]*rot[2]);
        for(int j=0;j<J;++j){f[6+j]=d->qpos[7+j];f[29+j]=.04*d->qvel[6+j];f[52+j]=.32*(prev[j]-neutral[j]);}
        f[75]=1.;
        double slew=0.;
        for(int j=0;j<J;++j){
          const float* g=gains+(k*J+j)*X;double base=0.;for(int a=0;a<X;++a)base+=g[a]*dx[a];
          double mu=0.;for(int a=0;a<76;++a)mu+=weights[a*J+j]*f[a];
          float sample=mu+sigma*noise[ix*J+j];
          latent[ix*J+j]=sample;
          double goal=goals[k*J+j]+std::clamp(base,-.65,.65);
          float act=static_cast<float>(4.*(goal-neutral[j]))+scale*std::tanh(sample);
          cmd[j]=static_cast<float>(neutral[j])+.25f*act;
          slew+=16.*(cmd[j]-prev[j])*(cmd[j]-prev[j]);
        }
        advance(m,d,lag+row*J,motor,cmd,prev,old);old=prev;prev=cmd;
        mj_kinematics(m,d);
        const double* R=d->xmat+9;
        double yaw=std::atan2(R[3],R[0]),C=std::cos(yaw),S=std::sin(yaw);
        float* a=amp+73*ix;
        for(int col:{0,2}){int off=col==0?0:3;a[off]=C*R[col]+S*R[3+col];a[off+1]=-S*R[col]+C*R[3+col];a[off+2]=R[6+col];}
        for(int j=0;j<3;++j)a[6+j]=d->qvel[3+j];
        for(int j=0;j<J;++j){a[9+j]=d->qpos[7+j];a[32+j]=d->qvel[6+j];}
        for(int j=0;j<6;++j)for(int k1=0;k1<3;++k1){double z=0.;for(int b=0;b<3;++b)z+=R[3*b+k1]*(d->xpos[links[j]*3+b]-d->qpos[b]);a[55+3*j+k1]=z;}
        mj_differentiatePos(m,dx,1.,q+30*(k+1),d->qpos);double qe=0.,ve=0.;
        for(int j=0;j<J;++j){qe+=dx[6+j]*dx[6+j];ve+=std::pow(d->qvel[6+j]-v[V*(k+1)+6+j],2);}
        bool up=d->qpos[2]>.58 && d->qpos[2]<.78 && R[8]>.866;
        if(up&&!rose){rose=true;anchor[0]=d->qpos[0];anchor[1]=d->qpos[1];}
        double dist=std::hypot(d->qpos[0]-anchor[0],d->qpos[1]-anchor[1]);
        double measured[2]={};
        for(int j=0;j<d->ncon;++j){
          const auto &co=d->contact[j];int a=m->geom_bodyid[co.geom1],b=m->geom_bodyid[co.geom2];if(a&&b)continue;
          for(int side=0;side<2;++side)if(a+b==links[4+side]){mjtNum cf[6];mj_contactForce(m,d,j,cf);measured[side]+=std::max(0.,cf[0])/weight;}
        }
        double arm_cost=0.,expected=0.,fast=0.;
        for(int side=0;side<2;++side){double ref=expected_hand[2*k+side];arm_cost+=std::pow(std::max(0.,ref-measured[side]),2);expected+=ref;}
        if(d->qpos[2]>.42 && R[8]>.53){
          for(int j:{0,3,6,9}){double e=std::max(0.,std::abs(d->qvel[6+j])-std::max(.70,std::abs(v[V*k+6+j])+.25));fast+=e*e;}
        }
        reward[ix]=.02*(-35.*arm_cost-.11*std::min(fast,30.)+4.*std::exp(-5.*qe/J)+2.*std::exp(-.08*ve/J)+3.*std::exp(-85*dx[2]*dx[2])
                   +up*(2.+6.*std::exp(-40*dist*dist-3.*std::pow(std::hypot(d->qvel[0],d->qvel[1]),2)))
                   -.015*slew -3.*(rose&&!up));
      }
    }
    mj_deleteData(d);
  }
}

extern "C" void stance(const mjModel*,mjData*,const int*,const double*,const double*,const double*,double,double,double*);

extern "C" void stance_control(const mjModel* m,mjData* d,const int* bodies,const double* motor,const double* neutral,
                               const double* anchors,const double* previous,const double* config,
                               const float* actor,double scale,float* commands) {
  double nominal[J],x[76];stance(m,d,bodies,motor,neutral,anchors,config[0],config[1],nominal);
  const double* quat=d->qpos+3;
  for(int j=0;j<3;++j)x[j]=.12*d->qvel[3+j];
  x[3]=2.*(quat[1]*quat[3]-quat[0]*quat[2]);x[4]=2.*(quat[2]*quat[3]+quat[0]*quat[1]);x[5]=1.-2.*(quat[1]*quat[1]+quat[2]*quat[2]);
  for(int j=0;j<J;++j){x[6+j]=d->qpos[7+j];x[29+j]=.04*d->qvel[6+j];x[52+j]=.32*(previous[j]-neutral[j]);}
  x[75]=1.;
  for(int j=0;j<J;++j){
    double mu=0.;for(int a=0;a<76;++a)mu+=x[a]*actor[a*J+j];
    commands[j]=static_cast<float>(nominal[j]+scale*std::tanh(mu));
  }
}

extern "C" int forecast_stance(const mjModel* m,mjData* d,int count,const int* lag,const int* bodies,
                               const double* motor,const double* neutral,const double* anchors,
                               const double* prev0,const double* old0,const double* config,const float* actor,double scale,
                               double* poses,double* speeds) {
  Motors previous,older,cmd;std::copy_n(prev0,J,previous.begin());std::copy_n(old0,J,older.begin());
  for(int k=0;k<count;++k){
    float u[J];stance_control(m,d,bodies,motor,neutral,anchors,previous.data(),config,actor,scale,u);
    for(int j=0;j<J;++j)cmd[j]=static_cast<float>(neutral[j])+.25f*u[j];
    advance(m,d,lag,motor,cmd,previous,older);older=previous;previous=cmd;
    std::copy_n(d->qpos,30,poses+k*30);std::copy_n(d->qvel,V,speeds+k*V);
    if(k>=18&&(d->qpos[2]<.40 || 1.-2.*(d->qpos[4]*d->qpos[4]+d->qpos[5]*d->qpos[5])<.48))return k+1;
  }
  return count;
}
