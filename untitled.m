g=9.81;%m/s^2
m=2;%kg
Iz=1/2*m*0.1^2;%kg*m^2
Ix=1/12*m*(3*0.1^2+0.1^2);
Iy=Ix;
A=[
    0,1,0,0,0,0,0,0,0,0,0,0;
    0,0,0,0,0,0,0,0,0,0,0,0;
    0,0,0,1,0,0,0,0,0,0,0,0;
    0,0,0,0,0,0,0,0,0,0,0,0;
    0,0,0,0,0,1,0,0,0,0,0,0;
    0,0,0,0,0,0,g,0,0,0,0,0;
    0,0,0,0,0,0,0,1,0,0,0,0;
    0,0,0,0,0,0,0,0,0,0,0,0;
    0,0,0,0,0,0,0,0,0,1,0,0;
    0,0,0,0,0,0,0,0,0,0,-g,0;
    0,0,0,0,0,0,0,0,0,0,0,1;
    0,0,0,0,0,0,0,0,0,0,0,0;
    ];

B=[
    0,0,0,0;
    1/m,0,0,0;
    0,0,0,0;
    0,1/Ix,0,0;
    0,0,0,0;
    0,0,0,0;
    0,0,0,0;
    0,0,1/Iy,0;
    0,0,0,0;
    0,0,0,0;
    0,0,0,0;
    0,0,0,1/Iz;
    ];


C=[
    1,0,0,0,0,0,0,0,0,0,0,0;
    0,1,0,0,0,0,0,0,0,0,0,0;
    0,0,1,0,0,0,0,0,0,0,0,0;
    0,0,0,0,0,0,1,0,0,0,0,0;
    0,0,0,0,0,0,0,1,0,0,0,0;
    0,0,0,0,0,0,0,0,1,0,0,0;
    ];
D=zeros(6,4);

sys=ss(A,B,C,D);
%x=      z    zd  psi psid    x   xd phi  phid    y   yd  theta   thetad
acceptable_error_states=[200,10,1,0.1,200,10,1,0.1,200,10,1,0.1];

Q=diag(1./(acceptable_error_states.^2));

acceptable_error_input=[30,10,10,10];
R=diag(1./acceptable_error_input.^2);
K=lqr(sys,Q,R);
sys_c=ss(A-B*K,B,C,D);

x=zeros(12,1);
x_des=zeros(12,1);
x_des(1)=-10;

tspan=0:0.01:30;
xs=zeros(12,length(tspan));
us=zeros(4,length(tspan));
locals=zeros(3,length(tspan));

target=[500,0,100]';%in x y z
des_dist=100;


x_des(5)=target(1)-des_dist;
x_des(9)=target(2);
x_des(1)=target(3);



for t=tspan
    u=K*(x_des-x);
    u(1)=max(min(u(1),100),0);
    u(2)=max(min(u(2),10),-10);
    u(3)=max(min(u(3),10),-10);
    u(4)=max(min(u(4),10),-10);
 
    xd=A*x+B*u;
    
    xd(2)=xd(2)-g;
    x=x+0.01*xd;
    us(:,int32(t*100+1))=u;
    xs(:,int32(t*100+1))=x;
    locals(:,int32(t*100+1))=inLocalView(x,target);
end

figure(1)
plot(us')
legend('throttle','pitch','roll','yaw')
figure(2)
plot(xs')
legend('z','zd','psi','psid','x','xd','phi','phid','y','yd','theta','thetad')
figure(3)
plot(locals')
legend('x','y','z')


function local=inLocalView(states,target)
    z=states(1);
    zd=states(2);
    yaw=states(3);
    psid=states(4);
    x=states(5);
    xd=states(6);
    roll=states(7);
    phid=states(8);
    y=states(9);
    yd=states(10);
    pitch=states(11);
    thetad=states(12);

    rel=target-[x;y;z];
    v=Rot(yaw,pitch,roll)*rel;
    xl=v(1);
    yl=v(2);
    zl=v(3);

    local=[xl;yl;zl];
    



end
function M=Rot(a,b,g)
M=[
    cos(a)*cos(b),cos(a)*sin(b)*sin(g)-sin(a)*cos(g),cos(a)*sin(b)*cos(g)+sin(a)*sin(g);
    sin(a)*cos(b),sin(a)*sin(b)*sin(g)+cos(a)*cos(g),sin(a)*sin(b)*cos(g)-cos(a)*sin(g);
    -sin(b),cos(b)*sin(g),cos(b)*cos(g)
    ];
end