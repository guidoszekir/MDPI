defaultNode = ros2node("/default_node")
pause(2)

experimento = 1 %Dados que seram salvos na matriz, que diferencia qual experimento esta sendo feito. 1 para UGV, 2 para UAV
numero_exp = 0 %Numero do experimento. 

subscriber_pose_armadilha_1 = ros2subscriber(defaultNode,"/pose_armadilha_1")
% subscriber_pose_armadilha_2 = ros2subscriber(defaultNode,"/pose_armadilha_2")
% subscriber_pose_armadilha_3 = ros2subscriber(defaultNode,"/pose_armadilha_3")
subscriber_pose_drone = ros2subscriber(defaultNode,"/pose_drone")
% subscriber_cmd_vel_drone = ros2subscriber(defaultNode,"/drone/cmd_vel")
% subscriber_pose_husk = ros2subscriber(defaultNode,"/pose_husk")
% subscriber_cmd_vel_husk = ros2subscriber(defaultNode,"/husky/cmd_vel")
pause(2)

%A posição da armadilha é fixa, pode ser coletada uma unica vez
pose_armadilha_1 = receive(subscriber_pose_armadilha_1)
% pose_armadilha_2 = receive(subscriber_pose_armadilha_2)
% pose_armadilha_3 = receive(subscriber_pose_armadilha_3)


%loop para coletar a posição do robô e do Drone, assim como as
%velocidades enviadas para o equipamento

if exist('Dados_UAV','var') == 0
    Dados_UAV = [] %lx,ly,lz,ax,ay,ax,vlx,vly,vlz,vax,vay,vaz;
end
if exist('Dados_UGV','var') == 0
    Dados_UGV = [] %lx,ly,lz,ax,ay,ax,vlx,vly,vlz,vax,vay,vaz;
end



if exist('i','var') == 0
    i = 1
else
    i = i+1
end

tempo_inicio = tic;


while true
    temp_pose = receive(subscriber_pose_drone);
    %temp_vel = receive(subscriber_cmd_vel_drone);
    Dados_UAV(i,1) = temp_pose.linear.x;
    Dados_UAV(i,2) = temp_pose.linear.y;
    Dados_UAV(i,3) = temp_pose.linear.z;
    Dados_UAV(i,4) = temp_pose.angular.x;
    Dados_UAV(i,5) = temp_pose.angular.y;
    Dados_UAV(i,6) = temp_pose.angular.z; 
    
    Dados_UAV(i,7) = experimento;
    Dados_UAV(i,8) = numero_exp;
    Dados_UAV(i,9) = toc(tempo_inicio) ;

    %Dados_UAV(i,7) = temp_vel.linear.x;
    %Dados_UAV(i,8) = temp_vel.linear.y;
    %Dados_UAV(i,9) = temp_vel.linear.z;
    %Dados_UAV(i,10) = temp_vel.angular.x;
    %Dados_UAV(i,11) = temp_vel.angular.y;
    %Dados_UAV(i,12) = temp_vel.angular.z;

%     temp_pose = receive(subscriber_pose_husk);
%     %temp_vel = receive(subscriber_cmd_vel_husk);
%     Dados_UGV(i,1) = temp_pose.linear.x;
%     Dados_UGV(i,2) = temp_pose.linear.y;
%     Dados_UGV(i,3) = temp_pose.linear.z;
%     Dados_UGV(i,4) = temp_pose.angular.x;
%     Dados_UGV(i,5) = temp_pose.angular.y;
%     Dados_UGV(i,6) = temp_pose.angular.z;
% 
%     Dados_UGV(i,7) = experimento;
%     Dados_UGV(i,8) = numero_exp;
%     Dados_UGV(i,9) = toc(tempo_inicio) ;

    %Dados_UGV(i,7) = temp_vel.linear.x;
    %Dados_UGV(i,8)= temp_vel.linear.y;
    %Dados_UGV(i,9) = temp_vel.linear.z;
    %Dados_UGV(i,10) = temp_vel.angular.x;
    %Dados_UGV(i,11) = temp_vel.angular.y;
    %Dados_UGV(i,12) = temp_vel.angular.z;
    pause(0.2);
    i = i+1;
end


