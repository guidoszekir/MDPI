defaultNode = ros2node("/default_node")
pause(2)


subscriber_YoLo = ros2subscriber(defaultNode,"/dados_YoLo_controle")
subscriber_inicio = ros2subscriber(defaultNode,"/YoLo_troca_imagem")
subscriber_dir = ros2subscriber(defaultNode,"/UGV_direcao")

pause(2)

pub_vel_UAV = ros2publisher(defaultNode,"/drone/cmd_vel","geometry_msgs/Twist");
velMsg = ros2message(pub_vel_UAV);


YoLo = receive(subscriber_YoLo)


while true
    inicio = receive(subscriber_inicio)
    if inicio.data == 2
        velMsg.linear.x = 0;
        velMsg.linear.z = 0.3;
        velMsg.angular.z = 0;
        send(pub_vel_UAV,velMsg);
        pause(2)
        velMsg.linear.x = 0;
        velMsg.linear.z = 0;
        velMsg.angular.z = 0;
        send(pub_vel_UAV,velMsg);

        dir = receive(subscriber_dir)
        %Depois remover
        dir.data = 2
        if dir.data == 1
            velMsg.linear.x = 0;
            velMsg.linear.z = 0;
            velMsg.angular.z = 0.3;
            send(pub_vel_UAV,velMsg);
            pause(2)
        elseif  dir.data ==2
            elMsg.linear.x = 0;
            velMsg.linear.z = 0;
            velMsg.angular.z = 0.3;
            send(pub_vel_UAV,velMsg);
            pause(1)
        elseif dir.data == 4
            velMsg.linear.x = 0;
            velMsg.linear.z = 0;
            velMsg.angular.z = -0.3;
            send(pub_vel_UAV,velMsg);
            pause(1)
        elseif  dir.data ==5
            elMsg.linear.x = 0;
            velMsg.linear.z = 0;
            velMsg.angular.z = -0.3;
            send(pub_vel_UAV,velMsg);
            pause(2)    

        end
        velMsg.linear.x = 0;
        velMsg.linear.z = 0;
        velMsg.angular.z = 0;
        send(pub_vel_UAV,velMsg);
        break
    end

end

while true
    %porcentagem do objeto pela imagem
    YoLo = receive(subscriber_YoLo);

    if YoLo.data(1) ~= 0 && YoLo.data(2) ~= 0 && YoLo.data(3) ~= 0 && YoLo.data(4) ~= 0

        porArmadilhaY = (double(YoLo.data(2))/480)*100;    
        saida_fuzzy = evalfis(fuzzy_UAV,[double(YoLo.data(3)) double(YoLo.data(4)) porArmadilhaY]);
        velMsg.linear.x = saida_fuzzy(3)*0.3;
        velMsg.linear.z = -(saida_fuzzy(1)*0.4);
        velMsg.angular.z = (saida_fuzzy(2)*0.1);
        send(pub_vel_UAV,velMsg);
        pause(0.3)
        porArmadilhaY
        velMsg.linear.z
    
        if porArmadilhaY > 70
            velMsg.linear.x = 0;
            velMsg.linear.z = 0;
            velMsg.angular.z = 0;
            send(pub_vel_UAV,velMsg);
            break
        end

    end

end