defaultNode = ros2node("/default_node")
pause(2)



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



if exist('i','var') == 0
    i = 1
end

experimento = 1 %Dados que seram salvos na matriz, que diferencia qual experimento esta sendo feito. 1 para UGV, 2 para UAV

if exist('DadosExp','var') == 0
    DadosExp = []
end    

temp_pose = receive(subscriber_pose_drone);
DadosExp(i,1) = sqrt((temp_pose.linear.x-pose_armadilha_1.linear.x)^2 + (temp_pose.linear.y-pose_armadilha_1.linear.y)^2 + (temp_pose.linear.z-pose_armadilha_1.linear.z)^2);
% DadosExp(i,2) = sqrt((temp_pose.linear.x-pose_armadilha_2.linear.x)^2 + (temp_pose.linear.y-pose_armadilha_2.linear.y)^2);
% DadosExp(i,3) = sqrt((temp_pose.linear.x-pose_armadilha_3.linear.x)^2 + (temp_pose.linear.y-pose_armadilha_3.linear.y)^2);
DadosExp(i,4) = 2


i = i+1;




