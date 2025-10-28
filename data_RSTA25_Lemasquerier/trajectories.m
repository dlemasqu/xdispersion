close all
clear all


traj=load('fort.1001'); %Load file with trajectories

np = 1000 ; %number of particles
nt = 200+1;  %number of iterations/time steps for each trajectory

%% Raw trajectories
figure('Position',[100,100,1000,1000]);
for k=1:np %loop on (some) particles. 
    xtraj = traj((k-1)*nt+1:k*nt,1);
    ytraj = traj((k-1)*nt+1:k*nt,2);
    plot(xtraj,ytraj,'-','LineWidth',1.2);
    hold on
end
axis equal
xlabel('$x$ (m)')
ylabel('$y$ (m)')


%% Movie motion particles


%Import velocity field for first time step
ngrid=201; %velocity field grid size
UV=readmatrix('ExpA_1000.dat', 'FileType', 'text', 'Range', [1, 1, ngrid^2, 4]);

figure('Position',[200,600,1200,1200]);
quiver(UV(:,1),UV(:,2),UV(:,3),UV(:,4),5,'color',0.8*[1,1,1]); hold on
%initial positions
xp0 = traj(1:nt:end,1);
yp0 = traj(1:nt:end,2);
rp0=sqrt(xp0.^2+yp0.^2); %initial radius of each particle 
for i=1:np
    xp=traj(i:nt:end,1);
    yp=traj(i:nt:end,2);
    ss=scatter(xp,yp,20,rp0,'linewidth',1.5) %scatter plot, colour based on initial radius 
    xlim([-0.5,0.5])
    ylim([-0.5,0.5])

    pause(0.05)
    delete(ss)
end


