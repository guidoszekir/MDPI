close all;
clear all;
clc;

% %%Graficos de superficie
fuzzyPosLider = readfis('fuzzy_UAV');
% % 
figure('Name','FCServo');
opt = gensurfOptions('OutputIndex',1);
opt.InputIndex = [1 2];
gensurf(fuzzyPosLider, opt)
title('');
set(findall(gcf,'type','text'),'Fontsize', 50)
set(gca,'Fontsize', 60)
set(findall(gca),'linewidth', 4) 
% % ylabel('box center j','Interpreter','tex','FontSize',60,'FontWeight','bold');
% zlabel('Velocidade Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
set(gca,'fontsize',30)
set(findall(gcf,'type','text'),'fontSize',40) 

figure('Name','FCServo');
opt = gensurfOptions('OutputIndex',2);
opt.InputIndex = [1 2];
gensurf(fuzzyPosLider, opt)
title('');
set(findall(gcf,'type','text'),'Fontsize', 50)
set(gca,'Fontsize', 60)
set(findall(gca),'linewidth', 4) 
% ylabel('Diferença Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% zlabel('Velocidade Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
set(gca,'fontsize',30)
set(findall(gcf,'type','text'),'fontSize',40) 

figure('Name','FCServo');
opt = gensurfOptions('OutputIndex',3);
opt.InputIndex = [1 2];
gensurf(fuzzyPosLider, opt)
title('');
set(findall(gcf,'type','text'),'Fontsize', 50)
set(gca,'Fontsize', 60)
set(findall(gca),'linewidth', 4) 
% ylabel('Diferença Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% zlabel('Velocidade Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
set(gca,'fontsize',30)
set(findall(gcf,'type','text'),'fontSize',40) 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
figure('Name','FCServo');
opt = gensurfOptions('OutputIndex',1);
opt.InputIndex = [1 3];
gensurf(fuzzyPosLider, opt)
title('');
set(findall(gcf,'type','text'),'Fontsize', 50)
set(gca,'Fontsize', 60)
set(findall(gca),'linewidth', 4) 
% ylabel('Diferença Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% zlabel('Velocidade Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
set(gca,'fontsize',30)
set(findall(gcf,'type','text'),'fontSize',40) 

figure('Name','FCServo');
opt = gensurfOptions('OutputIndex',2);
opt.InputIndex = [1 3];
gensurf(fuzzyPosLider, opt)
title('');
set(findall(gcf,'type','text'),'Fontsize', 50)
set(gca,'Fontsize', 60)
set(findall(gca),'linewidth', 4) 
% ylabel('Diferença Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% zlabel('Velocidade Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
set(gca,'fontsize',30)
set(findall(gcf,'type','text'),'fontSize',40) 

figure('Name','FCServo');
opt = gensurfOptions('OutputIndex',3);
opt.InputIndex = [1 3];
gensurf(fuzzyPosLider, opt)
title('');
set(findall(gcf,'type','text'),'Fontsize', 50)
set(gca,'Fontsize', 60)
set(findall(gca),'linewidth', 4) 
% ylabel('Diferença Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% zlabel('Velocidade Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
set(gca,'fontsize',30)
set(findall(gcf,'type','text'),'fontSize',40) 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Name','FCServo');
opt = gensurfOptions('OutputIndex',1);
opt.InputIndex = [2 3];
gensurf(fuzzyPosLider, opt)
title('');
set(findall(gcf,'type','text'),'Fontsize', 50)
set(gca,'Fontsize', 60)
set(findall(gca),'linewidth', 4) 
% ylabel('Diferença Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% zlabel('Velocidade Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
set(gca,'fontsize',30)
set(findall(gcf,'type','text'),'fontSize',40) 

figure('Name','FCServo');
opt = gensurfOptions('OutputIndex',2);
opt.InputIndex = [2 3];
gensurf(fuzzyPosLider, opt)
title('');
set(findall(gcf,'type','text'),'Fontsize', 50)
set(gca,'Fontsize', 60)
set(findall(gca),'linewidth', 4) 
% ylabel('Diferença Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% zlabel('Velocidade Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
set(gca,'fontsize',30)
set(findall(gcf,'type','text'),'fontSize',40) 

figure('Name','FCServo');
opt = gensurfOptions('OutputIndex',3);
opt.InputIndex = [2 3];
gensurf(fuzzyPosLider, opt)
title('');
set(findall(gcf,'type','text'),'Fontsize', 50)
set(gca,'Fontsize', 60)
set(findall(gca),'linewidth', 4) 
% ylabel('Diferença Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% zlabel('Velocidade Linear','Interpreter','tex','FontSize',60,'FontWeight','bold');
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
set(gca,'fontsize',30)
set(findall(gcf,'type','text'),'fontSize',40) 



