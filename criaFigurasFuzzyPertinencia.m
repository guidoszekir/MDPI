close all;
clear all;
clc;

% %%Graficos de superficie
fuzzyPosLider = readfis('fuzzy_UAV');

% %### Pertinencia
figure('Name','Input FCServo');
plotmf(fuzzyPosLider,'input',1);
title('');
set(findall(gcf,'type','text'),'Fontsize', 30,'FontWeight','bold')
set(gca,'Fontsize', 40, 'FontWeight','bold')
set(findall(gca),'linewidth', 7) 
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
%  ylabel('Grau','Interpreter','tex','FontSize',60,'FontWeight','bold');


% %### Pertinencia
figure('Name','Input FCServo');
plotmf(fuzzyPosLider,'input',2);
title('');
set(findall(gcf,'type','text'),'Fontsize', 30,'FontWeight','bold')
set(gca,'Fontsize', 40, 'FontWeight','bold')
set(findall(gca),'linewidth', 7) 
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
%  ylabel('Grau','Interpreter','tex','FontSize',60,'FontWeight','bold');

% %### Pertinencia
figure('Name','Input FCServo');
plotmf(fuzzyPosLider,'input',3);
title('');
set(findall(gcf,'type','text'),'Fontsize', 30,'FontWeight','bold')
set(gca,'Fontsize', 40, 'FontWeight','bold')
set(findall(gca),'linewidth', 7) 
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
%  ylabel('Grau','Interpreter','tex','FontSize',60,'FontWeight','bold');


% %### Pertinencia
figure('Name','Input FCServo');
plotmf(fuzzyPosLider,'output',1);
title('');
set(findall(gcf,'type','text'),'Fontsize', 30,'FontWeight','bold')
set(gca,'Fontsize', 40, 'FontWeight','bold')
set(findall(gca),'linewidth', 7) 
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
%  ylabel('Grau','Interpreter','tex','FontSize',60,'FontWeight','bold');


% %### Pertinencia
figure('Name','Input FCServo');
plotmf(fuzzyPosLider,'output',2);
title('');
set(findall(gcf,'type','text'),'Fontsize', 30,'FontWeight','bold')
set(gca,'Fontsize', 40, 'FontWeight','bold')
set(findall(gca),'linewidth', 7) 
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
%  ylabel('Grau','Interpreter','tex','FontSize',60,'FontWeight','bold');


% %### Pertinencia
figure('Name','Input FCServo');
plotmf(fuzzyPosLider,'output',3);
title('');
set(findall(gcf,'type','text'),'Fontsize', 30,'FontWeight','bold')
set(gca,'Fontsize', 40, 'FontWeight','bold')
set(findall(gca),'linewidth', 7) 
% xlabel('Diferença Angular','Interpreter','tex','FontSize',60,'FontWeight','bold');
%  ylabel('Grau','Interpreter','tex','FontSize',60,'FontWeight','bold');
