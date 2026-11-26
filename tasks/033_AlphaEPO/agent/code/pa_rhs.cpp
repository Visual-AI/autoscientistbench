#include <cmath>
double fun_e_uu_uu(double k,double p,double alpha,double q,double c_uu,double c_ud,double c_du,double c_dd,double theta_uu_uu,double theta_uu_ud,double theta_uu_du,double theta_uu_dd,double theta_ud_uu,double theta_ud_ud,double theta_ud_du,double theta_ud_dd,double theta_du_uu,double theta_du_ud,double theta_du_du,double theta_du_dd,double theta_dd_uu,double theta_dd_ud,double theta_dd_du,double theta_dd_dd){
double c1=((((((-c_uu)*p)*alpha)*theta_uu_uu)-(((((((2.0*c_uu)/k)*(1.0-p))*alpha)*theta_uu_uu)*pow((theta_uu_du+theta_uu_dd),q))*(k-q)))-(((((((2.0*c_uu)/k)*(1.0-p))*(1.0-alpha))*theta_uu_uu)*pow((theta_uu_du+theta_uu_dd),q))*(k-q)));
double c2=((((c_ud*theta_ud_uu)*p)*alpha)+(((((((2.0*c_ud)/k)*alpha)*(1.0-p))*theta_ud_uu)*pow((theta_ud_uu+theta_ud_ud),(q-1.0)))*(q+((k-q)*(theta_ud_uu+theta_ud_ud)))));
double c3=((((2.0*c_du)*(1.0-alpha))*theta_du_uu)-(((((((2.0*c_du)/k)*(1.0-alpha))*(1.0-p))*theta_du_uu)*pow((theta_du_du+theta_du_dd),q))*(k-q)));
double dydt_uu_uu=((c1+c2)+c3);
return dydt_uu_uu; }
double fun_e_uu_ud(double k,double p,double alpha,double q,double c_uu,double c_ud,double c_du,double c_dd,double theta_uu_uu,double theta_uu_ud,double theta_uu_du,double theta_uu_dd,double theta_ud_uu,double theta_ud_ud,double theta_ud_du,double theta_ud_dd,double theta_du_uu,double theta_du_ud,double theta_du_du,double theta_du_dd,double theta_dd_uu,double theta_dd_ud,double theta_dd_du,double theta_dd_dd){
double c1=(((((((((-c_uu)/k)*(1.0-p))*(1.0-alpha))*theta_uu_ud)*pow((theta_uu_du+theta_uu_dd),q))*(k-q))+(((((c_uu*1.0)/2.0)*p)*alpha)*(theta_uu_uu-theta_uu_ud)))+((((((c_uu/k)*alpha)*(1.0-p))*(theta_uu_uu-theta_uu_ud))*pow((theta_uu_du+theta_uu_dd),q))*(k-q)));
double c2=((((((-c_ud)*(1.0-alpha))*theta_ud_uu)+((((((c_ud/k)*(1.0-alpha))*(1.0-p))*theta_ud_uu)*pow((theta_ud_uu+theta_ud_ud),(q-1.0)))*(q+((k-q)*(theta_ud_uu+theta_ud_ud)))))+(((((c_ud*1.0)/2.0)*p)*alpha)*(theta_ud_ud-theta_ud_uu)))+((((((c_ud/k)*alpha)*(1.0-p))*(theta_ud_ud-theta_ud_uu))*pow((theta_ud_ud+theta_ud_uu),(q-1.0)))*(q+((k-q)*(theta_ud_ud+theta_ud_uu)))));
double c3=(((c_du*(1.0-alpha))*theta_du_ud)-((((((c_du/k)*(1.0-alpha))*(1.0-p))*theta_du_ud)*pow((theta_du_du+theta_du_dd),q))*(k-q)));
double c4=((((((c_dd/k)*(1.0-p))*(1.0-alpha))*theta_dd_uu)*pow((theta_dd_uu+theta_dd_ud),(q-1.0)))*(q+((k-q)*(theta_dd_uu+theta_dd_ud))));
double dydt_uu_ud=(((c1+c2)+c3)+c4);
return dydt_uu_ud; }
double fun_e_uu_du(double k,double p,double alpha,double q,double c_uu,double c_ud,double c_du,double c_dd,double theta_uu_uu,double theta_uu_ud,double theta_uu_du,double theta_uu_dd,double theta_ud_uu,double theta_ud_ud,double theta_ud_du,double theta_ud_dd,double theta_du_uu,double theta_du_ud,double theta_du_du,double theta_du_dd,double theta_dd_uu,double theta_dd_ud,double theta_dd_du,double theta_dd_dd){
double c1=((((((((-c_uu)*1.0)/2.0)*p)*alpha)*theta_uu_du)-((((((c_uu/k)*alpha)*(1.0-p))*theta_uu_du)*pow((theta_uu_du+theta_uu_dd),(q-1.0)))*(q+((k-q)*(theta_uu_du+theta_uu_dd)))))+((((c_uu/k)*(1.0-p))*(1.0-alpha))*((((-theta_uu_du)*pow((theta_uu_du+theta_uu_dd),(q-1.0)))*(q+((k-q)*(theta_uu_du+theta_uu_dd))))+((theta_uu_uu*pow((theta_uu_du+theta_uu_dd),q))*(k-q)))));
double c2=((((((c_ud*1.0)/2.0)*p)*alpha)*theta_ud_du)+((((((c_ud/k)*alpha)*(1.0-p))*theta_ud_du)*pow((theta_ud_uu+theta_ud_ud),q))*(k-q)));
double c3=(((((c_du*(1.0-alpha))*((-theta_du_uu)+theta_du_du))-((((c_du/k)*(1.0-alpha))*(1.0-p))*((((-theta_du_uu)*pow((theta_du_du+theta_du_dd),q))*(k-q))+((theta_du_du*pow((theta_du_du+theta_du_dd),(q-1.0)))*(q+((k-q)*(theta_du_du+theta_du_dd)))))))-(((((c_du*1.0)/2.0)*p)*alpha)*theta_du_uu))-((((((c_du/k)*alpha)*(1.0-p))*theta_du_uu)*pow((theta_du_du+theta_du_dd),q))*(k-q)));
double c4=((((((c_dd*1.0)/2.0)*p)*alpha)*theta_dd_uu)+((((((c_dd/k)*alpha)*(1.0-p))*theta_dd_uu)*pow((theta_dd_uu+theta_dd_ud),(q-1.0)))*(q+((k-q)*(theta_dd_uu+theta_dd_ud)))));
double dydt_uu_du=(((c1+c2)+c3)+c4);
return dydt_uu_du; }
double fun_e_uu_dd(double k,double p,double alpha,double q,double c_uu,double c_ud,double c_du,double c_dd,double theta_uu_uu,double theta_uu_ud,double theta_uu_du,double theta_uu_dd,double theta_ud_uu,double theta_ud_ud,double theta_ud_du,double theta_ud_dd,double theta_du_uu,double theta_du_ud,double theta_du_du,double theta_du_dd,double theta_dd_uu,double theta_dd_ud,double theta_dd_du,double theta_dd_dd){
double c1=((((((((-c_uu)*1.0)/2.0)*p)*alpha)*theta_uu_dd)-((((((c_uu/k)*(1.0-p))*alpha)*theta_uu_dd)*pow((theta_uu_du+theta_uu_dd),(q-1.0)))*(q+((k-q)*(theta_uu_du+theta_uu_dd)))))-((((((c_uu/k)*(1.0-p))*(1.0-alpha))*theta_uu_dd)*pow((theta_uu_du+theta_uu_dd),(q-1.0)))*(q+((k-q)*(theta_uu_du+theta_uu_dd)))));
double c2=((((((((c_ud*1.0)/2.0)*p)*alpha)*theta_ud_dd)+((((((c_ud/k)*alpha)*(1.0-p))*theta_ud_dd)*pow((theta_ud_uu+theta_ud_ud),q))*(k-q)))+((c_ud*(1.0-alpha))*theta_ud_uu))-((((((c_ud/k)*(1.0-alpha))*(1.0-p))*theta_ud_uu)*pow((theta_ud_uu+theta_ud_ud),(q-1.0)))*(q+((k-q)*(theta_ud_uu+theta_ud_ud)))));
double c3=(((((c_du*(1.0-alpha))*theta_du_dd)-((((((c_du/k)*(1.0-p))*(1.0-alpha))*theta_du_dd)*pow((theta_du_du+theta_du_dd),(q-1.0)))*(q+((k-q)*(theta_du_du+theta_du_dd)))))+(((((c_du*1.0)/2.0)*p)*alpha)*theta_du_uu))+((((((c_du/k)*alpha)*(1.0-p))*theta_du_uu)*pow((theta_du_du+theta_du_dd),q))*(k-q)));
double c4=((((((((-c_dd)*1.0)/2.0)*p)*alpha)*theta_dd_uu)-((((((c_dd/k)*alpha)*(1.0-p))*theta_dd_uu)*pow((theta_dd_uu+theta_dd_ud),(q-1.0)))*(q+((k-q)*(theta_dd_uu+theta_dd_ud)))))-((((((c_dd/k)*(1.0-alpha))*(1.0-p))*theta_dd_uu)*pow((theta_dd_uu+theta_dd_ud),(q-1.0)))*(q+((k-q)*(theta_dd_uu+theta_dd_ud)))));
double dydt_uu_dd=(((c1+c2)+c3)+c4);
return dydt_uu_dd; }
double fun_e_ud_ud(double k,double p,double alpha,double q,double c_uu,double c_ud,double c_du,double c_dd,double theta_uu_uu,double theta_uu_ud,double theta_uu_du,double theta_uu_dd,double theta_ud_uu,double theta_ud_ud,double theta_ud_du,double theta_ud_dd,double theta_du_uu,double theta_du_ud,double theta_du_du,double theta_du_dd,double theta_dd_uu,double theta_dd_ud,double theta_dd_du,double theta_dd_dd){
double c1=((((c_uu*p)*alpha)*theta_uu_ud)+(((((((2.0*c_uu)/k)*alpha)*(1.0-p))*theta_uu_ud)*pow((theta_uu_du+theta_uu_dd),q))*(k-q)));
double c2=(((((((-c_ud)*p)*alpha)*theta_ud_ud)-(((((((2.0*c_ud)/k)*alpha)*(1.0-p))*theta_ud_ud)*pow((theta_ud_uu+theta_ud_ud),(q-1.0)))*(q+((k-q)*(theta_ud_uu+theta_ud_ud)))))-(((2.0*c_ud)*(1.0-alpha))*theta_ud_ud))+(((((((2.0*c_ud)/k)*(1.0-p))*(1.0-alpha))*theta_ud_ud)*pow((theta_ud_uu+theta_ud_ud),(q-1.0)))*(q+((k-q)*(theta_ud_uu+theta_ud_ud)))));
double c4=(((((((2.0*c_dd)/k)*(1.0-p))*(1.0-alpha))*theta_dd_ud)*pow((theta_dd_uu+theta_dd_ud),(q-1.0)))*(q+((k-q)*(theta_dd_uu+theta_dd_ud))));
double dydt_ud_ud=((c1+c2)+c4);
return dydt_ud_ud; }
double fun_e_ud_du(double k,double p,double alpha,double q,double c_uu,double c_ud,double c_du,double c_dd,double theta_uu_uu,double theta_uu_ud,double theta_uu_du,double theta_uu_dd,double theta_ud_uu,double theta_ud_ud,double theta_ud_du,double theta_ud_dd,double theta_du_uu,double theta_du_ud,double theta_du_du,double theta_du_dd,double theta_dd_uu,double theta_dd_ud,double theta_dd_du,double theta_dd_dd){
double c1=(((((((c_uu*1.0)/2.0)*p)*alpha)*theta_uu_du)+((((((c_uu/k)*(1.0-p))*alpha)*theta_uu_du)*pow((theta_uu_du+theta_uu_dd),(q-1.0)))*(q+((k-q)*(theta_uu_du+theta_uu_dd)))))+((((((c_uu/k)*(1.0-p))*(1.0-alpha))*theta_uu_ud)*pow((theta_uu_du+theta_uu_dd),q))*(k-q)));
double c2=(((((((((-c_ud)*1.0)/2.0)*p)*alpha)*theta_ud_du)-((((((c_ud/k)*alpha)*(1.0-p))*theta_ud_du)*pow((theta_ud_uu+theta_ud_ud),q))*(k-q)))-((c_ud*(1.0-alpha))*theta_ud_du))+((((((c_ud/k)*(1.0-alpha))*(1.0-p))*theta_ud_du)*pow((theta_ud_uu+theta_ud_ud),q))*(k-q)));
double c3=((((((-c_du)*(1.0-alpha))*theta_du_ud)+((((((c_du/k)*(1.0-alpha))*(1.0-p))*theta_du_ud)*pow((theta_du_du+theta_du_dd),q))*(k-q)))-(((((c_du*1.0)/2.0)*p)*alpha)*theta_du_ud))-((((((c_du/k)*alpha)*(1.0-p))*theta_du_ud)*pow((theta_du_du+theta_du_dd),q))*(k-q)));
double c4=((((((((c_dd/k)*(1.0-p))*(1.0-alpha))*theta_dd_du)*pow((theta_dd_ud+theta_dd_uu),q))*(k-q))+(((((c_dd*1.0)/2.0)*p)*alpha)*theta_dd_ud))+((((((c_dd/k)*alpha)*(1.0-p))*theta_dd_ud)*pow((theta_dd_uu+theta_dd_ud),(q-1.0)))*(q+((k-q)*(theta_dd_uu+theta_dd_ud)))));
double dydt_ud_du=(((c1+c2)+c3)+c4);
return dydt_ud_du; }
double fun_e_dd_ud(double k,double p,double alpha,double q,double c_uu,double c_ud,double c_du,double c_dd,double theta_uu_uu,double theta_uu_ud,double theta_uu_du,double theta_uu_dd,double theta_ud_uu,double theta_ud_ud,double theta_ud_du,double theta_ud_dd,double theta_du_uu,double theta_du_ud,double theta_du_du,double theta_du_dd,double theta_dd_uu,double theta_dd_ud,double theta_dd_du,double theta_dd_dd){
double c1=((((((((-c_dd)*1.0)/2.0)*p)*alpha)*theta_dd_ud)-((((((c_dd/k)*alpha)*(1.0-p))*theta_dd_ud)*pow((theta_dd_uu+theta_dd_ud),(q-1.0)))*(q+((k-q)*(theta_dd_uu+theta_dd_ud)))))+((((c_dd/k)*(1.0-p))*(1.0-alpha))*((((-theta_dd_ud)*pow((theta_dd_ud+theta_dd_uu),(q-1.0)))*(q+((k-q)*(theta_dd_ud+theta_dd_uu))))+((theta_dd_dd*pow((theta_dd_ud+theta_dd_uu),q))*(k-q)))));
double c2=((((((c_du*1.0)/2.0)*p)*alpha)*theta_du_ud)+((((((c_du/k)*alpha)*(1.0-p))*theta_du_ud)*pow((theta_du_dd+theta_du_du),q))*(k-q)));
double c3=(((((c_ud*(1.0-alpha))*((-theta_ud_dd)+theta_ud_ud))-((((c_ud/k)*(1.0-alpha))*(1.0-p))*((((-theta_ud_dd)*pow((theta_ud_ud+theta_ud_uu),q))*(k-q))+((theta_ud_ud*pow((theta_ud_ud+theta_ud_uu),(q-1.0)))*(q+((k-q)*(theta_ud_ud+theta_ud_uu)))))))-(((((c_ud*1.0)/2.0)*p)*alpha)*theta_ud_dd))-((((((c_ud/k)*alpha)*(1.0-p))*theta_ud_dd)*pow((theta_ud_ud+theta_ud_uu),q))*(k-q)));
double c4=((((((c_uu*1.0)/2.0)*p)*alpha)*theta_uu_dd)+((((((c_uu/k)*(1.0-p))*alpha)*theta_uu_dd)*pow((theta_uu_du+theta_uu_dd),(q-1.0)))*(q+((k-q)*(theta_uu_du+theta_uu_dd)))));
double dydt_dd_ud=(((c1+c2)+c3)+c4);
return dydt_dd_ud; }
double fun_e_du_du(double k,double p,double alpha,double q,double c_uu,double c_ud,double c_du,double c_dd,double theta_uu_uu,double theta_uu_ud,double theta_uu_du,double theta_uu_dd,double theta_ud_uu,double theta_ud_ud,double theta_ud_du,double theta_ud_dd,double theta_du_uu,double theta_du_ud,double theta_du_du,double theta_du_dd,double theta_dd_uu,double theta_dd_ud,double theta_dd_du,double theta_dd_dd){
double c4=((((c_dd*p)*alpha)*theta_dd_du)+(((((((2.0*c_dd)/k)*alpha)*(1.0-p))*theta_dd_du)*pow((theta_dd_uu+theta_dd_ud),q))*(k-q)));
double c3=(((((((-c_du)*p)*alpha)*theta_du_du)-(((((((2.0*c_du)/k)*alpha)*(1.0-p))*theta_du_du)*pow((theta_du_dd+theta_du_du),(q-1.0)))*(q+((k-q)*(theta_du_dd+theta_du_du)))))-(((2.0*c_du)*(1.0-alpha))*theta_du_du))+(((((((2.0*c_du)/k)*(1.0-p))*(1.0-alpha))*theta_du_du)*pow((theta_du_dd+theta_du_du),(q-1.0)))*(q+((k-q)*(theta_du_dd+theta_du_du)))));
double c1=(((((((2.0*c_uu)/k)*(1.0-p))*(1.0-alpha))*theta_uu_du)*pow((theta_uu_dd+theta_uu_du),(q-1.0)))*(q+((k-q)*(theta_uu_dd+theta_uu_du))));
double dydt_du_du=((c1+c3)+c4);
return dydt_du_du; }
double fun_e_dd_du(double k,double p,double alpha,double q,double c_uu,double c_ud,double c_du,double c_dd,double theta_uu_uu,double theta_uu_ud,double theta_uu_du,double theta_uu_dd,double theta_ud_uu,double theta_ud_ud,double theta_ud_du,double theta_ud_dd,double theta_du_uu,double theta_du_ud,double theta_du_du,double theta_du_dd,double theta_dd_uu,double theta_dd_ud,double theta_dd_du,double theta_dd_dd){
double c1=(((((((((-c_dd)/k)*(1.0-p))*(1.0-alpha))*theta_dd_du)*pow((theta_dd_ud+theta_dd_uu),q))*(k-q))+(((((c_dd*1.0)/2.0)*p)*alpha)*(theta_dd_dd-theta_dd_du)))+((((((c_dd/k)*alpha)*(1.0-p))*(theta_dd_dd-theta_dd_du))*pow((theta_dd_uu+theta_dd_ud),q))*(k-q)));
double c2=((((((-c_du)*(1.0-alpha))*theta_du_dd)+((((((c_du/k)*(1.0-alpha))*(1.0-p))*theta_du_dd)*pow((theta_du_dd+theta_du_du),(q-1.0)))*(q+((k-q)*(theta_du_dd+theta_du_du)))))+(((((c_du*1.0)/2.0)*p)*alpha)*(theta_du_du-theta_du_dd)))+((((((c_du/k)*alpha)*(1.0-p))*(theta_du_du-theta_du_dd))*pow((theta_du_du+theta_du_dd),(q-1.0)))*(q+((k-q)*(theta_du_du+theta_du_dd)))));
double c3=(((c_ud*(1.0-alpha))*theta_ud_du)-((((((c_ud/k)*(1.0-alpha))*(1.0-p))*theta_ud_du)*pow((theta_ud_uu+theta_ud_ud),q))*(k-q)));
double c4=((((((c_uu/k)*(1.0-p))*(1.0-alpha))*theta_uu_dd)*pow((theta_uu_dd+theta_uu_du),(q-1.0)))*(q+((k-q)*(theta_uu_dd+theta_uu_du))));
double dydt_dd_du=(((c1+c2)+c3)+c4);
return dydt_dd_du; }
double fun_e_dd_dd(double k,double p,double alpha,double q,double c_uu,double c_ud,double c_du,double c_dd,double theta_uu_uu,double theta_uu_ud,double theta_uu_du,double theta_uu_dd,double theta_ud_uu,double theta_ud_ud,double theta_ud_du,double theta_ud_dd,double theta_du_uu,double theta_du_ud,double theta_du_du,double theta_du_dd,double theta_dd_uu,double theta_dd_ud,double theta_dd_du,double theta_dd_dd){
double c4=((((((-c_dd)*p)*alpha)*theta_dd_dd)-(((((((2.0*c_dd)/k)*(1.0-p))*(1.0-alpha))*theta_dd_dd)*pow((theta_dd_ud+theta_dd_uu),q))*(k-q)))-(((((((2.0*c_dd)/k)*(1.0-p))*alpha)*theta_dd_dd)*pow((theta_dd_ud+theta_dd_uu),q))*(k-q)));
double c3=((((c_du*theta_du_dd)*p)*alpha)+(((((((2.0*c_du)/k)*alpha)*(1.0-p))*theta_du_dd)*pow((theta_du_dd+theta_du_du),(q-1.0)))*(q+((k-q)*(theta_du_dd+theta_du_du)))));
double c2=((((2.0*c_ud)*(1.0-alpha))*theta_ud_dd)-(((((((2.0*c_ud)/k)*(1.0-alpha))*(1.0-p))*theta_ud_dd)*pow((theta_ud_uu+theta_ud_ud),q))*(k-q)));
double dydt_dd_dd=((c2+c3)+c4);
return dydt_dd_dd; }
extern "C" void rhs(const double* y, double* out, double k, double p, double alpha, double q){
double c_uu=y[0]+y[1]+y[2]+y[3];
double c_ud=y[1]+y[4]+y[5]+y[6];
double c_du=y[2]+y[5]+y[7]+y[8];
double c_dd=y[3]+y[6]+y[8]+y[9];
double theta_uu_uu= c_uu != 0 ? y[0]/c_uu : 0;
double theta_uu_ud= c_uu != 0 ? y[1]/c_uu : 0;
double theta_uu_du= c_uu != 0 ? y[2]/c_uu : 0;
double theta_uu_dd= c_uu != 0 ? y[3]/c_uu : 0;
double theta_ud_uu= c_ud != 0 ? y[1]/c_ud : 0;
double theta_ud_ud= c_ud != 0 ? y[4]/c_ud : 0;
double theta_ud_du= c_ud != 0 ? y[5]/c_ud : 0;
double theta_ud_dd= c_ud != 0 ? y[6]/c_ud : 0;
double theta_du_uu= c_du != 0 ? y[2]/c_du : 0;
double theta_du_ud= c_du != 0 ? y[5]/c_du : 0;
double theta_du_du= c_du != 0 ? y[7]/c_du : 0;
double theta_du_dd= c_du != 0 ? y[8]/c_du : 0;
double theta_dd_uu= c_dd != 0 ? y[3]/c_dd : 0;
double theta_dd_ud= c_dd != 0 ? y[6]/c_dd : 0;
double theta_dd_du= c_dd != 0 ? y[8]/c_dd : 0;
double theta_dd_dd= c_dd != 0 ? y[9]/c_dd : 0;
out[0]=fun_e_uu_uu(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd);
out[1]=fun_e_uu_ud(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd);
out[2]=fun_e_uu_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd);
out[3]=fun_e_uu_dd(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd);
out[4]=fun_e_ud_ud(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd);
out[5]=fun_e_ud_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd);
out[6]=fun_e_dd_ud(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd);
out[7]=fun_e_du_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd);
out[8]=fun_e_dd_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd);
out[9]=fun_e_dd_dd(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd);
}
