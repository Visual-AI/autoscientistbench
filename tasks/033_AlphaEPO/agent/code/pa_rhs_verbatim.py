# Generated from PA_authors.m: only MATLAB syntax is translated.
def fun_e_uu_uu(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd):
    c1=-c_uu*p*alpha*theta_uu_uu -2*c_uu/k*(1-p)*alpha*theta_uu_uu*(theta_uu_du + theta_uu_dd)**q*(k-q)- 2*c_uu/k*(1-p)*(1-alpha)*theta_uu_uu*(theta_uu_du + theta_uu_dd)**q*(k-q)
    c2=c_ud*theta_ud_uu*p*alpha + 2*c_ud/k*alpha*(1-p)*theta_ud_uu*(theta_ud_uu + theta_ud_ud)**(q-1)*(q + (k-q)*(theta_ud_uu + theta_ud_ud))
    c3=2*c_du*(1-alpha)*theta_du_uu - 2*c_du/k*(1-alpha)*(1-p)*theta_du_uu*(theta_du_du + theta_du_dd)**q*(k-q)
    dydt_uu_uu=c1 + c2 + c3
    return dydt_uu_uu
def fun_e_uu_ud(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd):
    c1=-c_uu/k*(1-p)*(1-alpha)*theta_uu_ud*(theta_uu_du + theta_uu_dd)**q*(k-q) + c_uu*1/2*p*alpha*(theta_uu_uu - theta_uu_ud) + c_uu/k*alpha*(1-p)*(theta_uu_uu - theta_uu_ud)*(theta_uu_du + theta_uu_dd)**q*(k-q)
    c2=(-c_ud*(1-alpha)*theta_ud_uu + c_ud/k*(1-alpha)*(1-p)*theta_ud_uu*(theta_ud_uu + theta_ud_ud)**(q-1)*(q + (k-q)*(theta_ud_uu + theta_ud_ud)) + c_ud*1/2*p*alpha*(theta_ud_ud - theta_ud_uu) + c_ud/k*alpha*(1-p)*(theta_ud_ud - theta_ud_uu)*(theta_ud_ud + theta_ud_uu)**(q-1)*(q + (k-q)*(theta_ud_ud + theta_ud_uu)))
    c3=c_du*(1-alpha)*theta_du_ud - c_du/k*(1-alpha)*(1-p)*theta_du_ud*(theta_du_du + theta_du_dd)**q*(k-q)
    c4=c_dd/k*(1-p)*(1-alpha)*theta_dd_uu*(theta_dd_uu + theta_dd_ud)**(q-1)*(q + (k-q)*(theta_dd_uu + theta_dd_ud))
    dydt_uu_ud=c1 + c2 + c3 + c4
    return dydt_uu_ud
def fun_e_uu_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd):
    c1=-c_uu*1/2*p*alpha*theta_uu_du - c_uu/k*alpha*(1-p)*theta_uu_du*(theta_uu_du + theta_uu_dd)**(q-1)*(q + (k-q)*(theta_uu_du + theta_uu_dd)) + c_uu/k*(1-p)*(1-alpha)*(-theta_uu_du*(theta_uu_du + theta_uu_dd)**(q-1)*(q + (k-q)*(theta_uu_du + theta_uu_dd)) + theta_uu_uu*(theta_uu_du + theta_uu_dd)**q*(k-q))
    c2=c_ud*1/2*p*alpha*theta_ud_du + c_ud/k*alpha*(1-p)*theta_ud_du*(theta_ud_uu + theta_ud_ud)**q*(k-q)
    c3=(c_du*(1-alpha)*(-theta_du_uu + theta_du_du) - c_du/k*(1-alpha)*(1-p)*(-theta_du_uu*(theta_du_du + theta_du_dd)**q*(k-q) + theta_du_du*(theta_du_du + theta_du_dd)**(q-1)*(q + (k-q)*(theta_du_du + theta_du_dd))) - c_du*1/2*p*alpha*theta_du_uu - c_du/k*alpha*(1-p)*theta_du_uu*(theta_du_du + theta_du_dd)**q*(k-q))
    c4=c_dd*1/2*p*alpha*theta_dd_uu + c_dd/k*alpha*(1-p)*theta_dd_uu*(theta_dd_uu+ theta_dd_ud)**(q-1)*(q + (k-q)*(theta_dd_uu+ theta_dd_ud))
    dydt_uu_du=c1 + c2 + c3 + c4
    return dydt_uu_du
def fun_e_uu_dd(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd):
    c1=- c_uu*1/2*p*alpha*theta_uu_dd- c_uu/k*(1-p)*alpha*theta_uu_dd*(theta_uu_du + theta_uu_dd)**(q-1)*(q + (k-q)*(theta_uu_du + theta_uu_dd)) - c_uu/k*(1-p)*(1-alpha)*theta_uu_dd*(theta_uu_du + theta_uu_dd)**(q-1)*(q + (k-q)*(theta_uu_du + theta_uu_dd))
    c2=(c_ud*1/2*p*alpha*theta_ud_dd + c_ud/k*alpha*(1-p)*theta_ud_dd*(theta_ud_uu + theta_ud_ud)**q*(k-q) + c_ud*(1-alpha)*theta_ud_uu - c_ud/k*(1-alpha)*(1-p)*theta_ud_uu*(theta_ud_uu + theta_ud_ud)**(q-1)*(q + (k-q)*(theta_ud_uu + theta_ud_ud)))
    c3=(c_du*(1-alpha)*theta_du_dd - c_du/k*(1-p)*(1-alpha)*theta_du_dd*(theta_du_du + theta_du_dd)**(q-1)*(q + (k-q)*(theta_du_du + theta_du_dd)) + c_du*1/2*p*alpha*theta_du_uu + c_du/k*alpha*(1-p)*theta_du_uu*(theta_du_du + theta_du_dd)**q*(k-q))
    c4=-c_dd*1/2*p*alpha*theta_dd_uu- c_dd/k*alpha*(1-p)*theta_dd_uu*(theta_dd_uu + theta_dd_ud)**(q-1)*(q + (k-q)*(theta_dd_uu + theta_dd_ud)) - c_dd/k*(1-alpha)*(1-p)*theta_dd_uu*(theta_dd_uu + theta_dd_ud)**(q-1)*(q + (k-q)*(theta_dd_uu + theta_dd_ud))
    dydt_uu_dd=c1 + c2 + c3 + c4
    return dydt_uu_dd
def fun_e_ud_ud(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd):
    c1=c_uu*p*alpha*theta_uu_ud + 2*c_uu/k*alpha*(1-p)*theta_uu_ud*(theta_uu_du + theta_uu_dd)**q*(k-q)
    c2=(-c_ud*p*alpha*theta_ud_ud - 2*c_ud/k*alpha*(1-p)*theta_ud_ud*(theta_ud_uu + theta_ud_ud)**(q-1)*(q + (k-q)*(theta_ud_uu + theta_ud_ud)) - 2*c_ud*(1-alpha)*theta_ud_ud + 2*c_ud/k*(1-p)*(1-alpha)*theta_ud_ud*(theta_ud_uu + theta_ud_ud)**(q-1)*(q + (k-q)*(theta_ud_uu + theta_ud_ud)))
    c4=2*c_dd/k*(1-p)*(1-alpha)*theta_dd_ud*(theta_dd_uu + theta_dd_ud)**(q-1)*(q + (k-q)*(theta_dd_uu + theta_dd_ud))
    dydt_ud_ud=c1 + c2 + c4
    return dydt_ud_ud
def fun_e_ud_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd):
    c1=c_uu*1/2*p*alpha*theta_uu_du + c_uu/k*(1-p)*alpha*theta_uu_du*(theta_uu_du + theta_uu_dd)**(q-1)*(q + (k-q)*(theta_uu_du + theta_uu_dd)) + c_uu/k*(1-p)*(1-alpha)*theta_uu_ud*(theta_uu_du + theta_uu_dd)**q*(k-q)
    c2=(-c_ud*1/2*p*alpha*theta_ud_du - c_ud/k*alpha*(1-p)*theta_ud_du*(theta_ud_uu + theta_ud_ud)**q*(k-q) - c_ud*(1-alpha)*theta_ud_du + c_ud/k*(1-alpha)*(1-p)*theta_ud_du*(theta_ud_uu + theta_ud_ud)**q*(k-q))
    c3=(-c_du*(1-alpha)*theta_du_ud + c_du/k*(1-alpha)*(1-p)*theta_du_ud*(theta_du_du + theta_du_dd)**q*(k-q) - c_du*1/2*p*alpha*theta_du_ud - c_du/k*alpha*(1-p)*theta_du_ud*(theta_du_du + theta_du_dd)**q*(k-q))
    c4=c_dd/k*(1-p)*(1-alpha)*theta_dd_du*(theta_dd_ud + theta_dd_uu)**q*(k-q) + c_dd*1/2*p*alpha*theta_dd_ud + c_dd/k*alpha*(1-p)*theta_dd_ud*(theta_dd_uu + theta_dd_ud)**(q-1)*(q + (k-q)*(theta_dd_uu + theta_dd_ud))
    dydt_ud_du=c1 + c2 + c3 + c4
    return dydt_ud_du
def fun_e_dd_ud(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd):
    c1=-c_dd*1/2*p*alpha*theta_dd_ud - c_dd/k*alpha*(1-p)*theta_dd_ud*(theta_dd_uu + theta_dd_ud)**(q-1)*(q + (k-q)*(theta_dd_uu + theta_dd_ud)) + c_dd/k*(1-p)*(1-alpha)*(-theta_dd_ud*(theta_dd_ud + theta_dd_uu)**(q-1)*(q + (k-q)*(theta_dd_ud + theta_dd_uu)) + theta_dd_dd*(theta_dd_ud + theta_dd_uu)**q*(k-q))
    c2=c_du*1/2*p*alpha*theta_du_ud + c_du/k*alpha*(1-p)*theta_du_ud*(theta_du_dd + theta_du_du)**q*(k-q)
    c3=(c_ud*(1-alpha)*(-theta_ud_dd + theta_ud_ud) - c_ud/k*(1-alpha)*(1-p)*(-theta_ud_dd*(theta_ud_ud + theta_ud_uu)**q*(k-q) + theta_ud_ud*(theta_ud_ud + theta_ud_uu)**(q-1)*(q + (k-q)*(theta_ud_ud + theta_ud_uu))) - c_ud*1/2*p*alpha*theta_ud_dd - c_ud/k*alpha*(1-p)*theta_ud_dd*(theta_ud_ud + theta_ud_uu)**q*(k-q))
    c4=c_uu*1/2*p*alpha*theta_uu_dd + c_uu/k*(1-p)*alpha*theta_uu_dd*(theta_uu_du + theta_uu_dd)**(q-1)*(q + (k-q)*(theta_uu_du + theta_uu_dd))
    dydt_dd_ud=c1 + c2 + c3 + c4
    return dydt_dd_ud
def fun_e_du_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd):
    c4=c_dd*p*alpha*theta_dd_du + 2*c_dd/k*alpha*(1-p)*theta_dd_du*(theta_dd_uu + theta_dd_ud)**q*(k-q)
    c3=(-c_du*p*alpha*theta_du_du - 2*c_du/k*alpha*(1-p)*theta_du_du*(theta_du_dd + theta_du_du)**(q-1)*(q + (k-q)*(theta_du_dd + theta_du_du)) - 2*c_du*(1-alpha)*theta_du_du + 2*c_du/k*(1-p)*(1-alpha)*theta_du_du*(theta_du_dd + theta_du_du)**(q-1)*(q + (k-q)*(theta_du_dd + theta_du_du)))
    c1=2*c_uu/k*(1-p)*(1-alpha)*theta_uu_du*(theta_uu_dd + theta_uu_du)**(q-1)*(q + (k-q)*(theta_uu_dd + theta_uu_du))
    dydt_du_du=c1 + c3 + c4
    return dydt_du_du
def fun_e_dd_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd):
    c1=-c_dd/k*(1-p)*(1-alpha)*theta_dd_du*(theta_dd_ud + theta_dd_uu)**q*(k-q) + c_dd*1/2*p*alpha*(theta_dd_dd - theta_dd_du) + c_dd/k*alpha*(1-p)*(theta_dd_dd - theta_dd_du)*(theta_dd_uu + theta_dd_ud)**q*(k-q)
    c2=(-c_du*(1-alpha)*theta_du_dd + c_du/k*(1-alpha)*(1-p)*theta_du_dd*(theta_du_dd + theta_du_du)**(q-1)*(q + (k-q)*(theta_du_dd + theta_du_du)) + c_du*1/2*p*alpha*(theta_du_du - theta_du_dd) + c_du/k*alpha*(1-p)*(theta_du_du - theta_du_dd)*(theta_du_du + theta_du_dd)**(q-1)*(q + (k-q)*(theta_du_du + theta_du_dd)))
    c3=c_ud*(1-alpha)*theta_ud_du - c_ud/k*(1-alpha)*(1-p)*theta_ud_du*(theta_ud_uu + theta_ud_ud)**q*(k-q)
    c4=c_uu/k*(1-p)*(1-alpha)*theta_uu_dd*(theta_uu_dd + theta_uu_du)**(q-1)*(q + (k-q)*(theta_uu_dd + theta_uu_du))
    dydt_dd_du=c1 + c2 + c3 + c4
    return dydt_dd_du
def fun_e_dd_dd(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd):
    c4=-c_dd*p*alpha*theta_dd_dd - 2*c_dd/k*(1-p)*(1-alpha)*theta_dd_dd*(theta_dd_ud + theta_dd_uu)**q*(k-q) - 2*c_dd/k*(1-p)*alpha*theta_dd_dd*(theta_dd_ud + theta_dd_uu)**q*(k-q)
    c3=c_du*theta_du_dd*p*alpha + 2*c_du/k*alpha*(1-p)*theta_du_dd*(theta_du_dd + theta_du_du)**(q-1)*(q + (k-q)*(theta_du_dd + theta_du_du))
    c2=2*c_ud*(1-alpha)*theta_ud_dd - 2*c_ud/k*(1-alpha)*(1-p)*theta_ud_dd*(theta_ud_uu + theta_ud_ud)**q*(k-q)
    dydt_dd_dd=c2 + c3 + c4
    return dydt_dd_dd
def rhs(y,k,p,alpha,q):
    c_uu=y[0]+y[1]+y[2]+y[3]
    c_ud=y[1]+y[4]+y[5]+y[6]
    c_du=y[2]+y[5]+y[7]+y[8]
    c_dd=y[3]+y[6]+y[8]+y[9]
    theta_uu_uu= y[0]/c_uu if c_uu != 0 else 0
    theta_uu_ud= y[1]/c_uu if c_uu != 0 else 0
    theta_uu_du= y[2]/c_uu if c_uu != 0 else 0
    theta_uu_dd= y[3]/c_uu if c_uu != 0 else 0
    theta_ud_uu= y[1]/c_ud if c_ud != 0 else 0
    theta_ud_ud= y[4]/c_ud if c_ud != 0 else 0
    theta_ud_du= y[5]/c_ud if c_ud != 0 else 0
    theta_ud_dd= y[6]/c_ud if c_ud != 0 else 0
    theta_du_uu= y[2]/c_du if c_du != 0 else 0
    theta_du_ud= y[5]/c_du if c_du != 0 else 0
    theta_du_du= y[7]/c_du if c_du != 0 else 0
    theta_du_dd= y[8]/c_du if c_du != 0 else 0
    theta_dd_uu= y[3]/c_dd if c_dd != 0 else 0
    theta_dd_ud= y[6]/c_dd if c_dd != 0 else 0
    theta_dd_du= y[8]/c_dd if c_dd != 0 else 0
    theta_dd_dd= y[9]/c_dd if c_dd != 0 else 0
    return [fun_e_uu_uu(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd),fun_e_uu_ud(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd),fun_e_uu_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd),fun_e_uu_dd(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd),fun_e_ud_ud(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd),fun_e_ud_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd),fun_e_dd_ud(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd),fun_e_du_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd),fun_e_dd_du(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd),fun_e_dd_dd(k,p,alpha,q,c_uu,c_ud,c_du,c_dd,theta_uu_uu,theta_uu_ud,theta_uu_du,theta_uu_dd,theta_ud_uu,theta_ud_ud,theta_ud_du,theta_ud_dd,theta_du_uu,theta_du_ud,theta_du_du,theta_du_dd,theta_dd_uu,theta_dd_ud,theta_dd_du,theta_dd_dd)]
