#!/bin/bash
alpha_length=4
csd_length=3
epoch_length=5
times=5
hp='--data=cifar10 --n_round=20 --n_client=20 --pruing_p=0'
alpha=('0.01' '0.05' '0.1' '1')
n_epoch='160'
gpu_num=('0' '1' '2' '3')
time_num='0' #('0' '1' '2' '3' '4')
paras[0]='./cnn_lab/cnn_fedavg.py'
paras[1]='./cnn_lab/cnn_ours.py'
paras[2]='./cnn_lab/cnn_curv.py'
paras[3]='./cnn_lab/cnn_prox.py'
py_number=4
csd_ours=('100' '1000' '10000')
#'0' '1' '10' '100' '1000' '10000'
csd_curv=('100' '1000' '10000')
csd_prox=('0.001' '0.0001' '1')
#'0' '0.1' '0.01' '0.001' '0.0001' '1'

for((a=0; a<alpha_length; a++))
    do
        {
        for((t=0; t<times; t++))
            do
                {
                 for((p=0; p<py_number; p++))
                        do
                            {
                                if [ $p == 0 ]
                                then
                                   echo "Run AVG No Csd"
                                   # python ${paras[p]} ${hp} '--n_epoch='${n_epoch[e]} '--alpha='${alpha[a]} '--csd_importance=0' '--gpu='${gpu_num[a]}
                                elif [ $p == 1 ]
                                then
                                   echo "Run Ours with Csd"
                                   for((c=0; c<csd_length; c++))
                                        do
                                            {
                                                python ${paras[p]} ${hp} '--n_epoch='${n_epoch} '--alpha='${alpha[a]} '--csd_importance='${csd_ours[c]} '--gpu='${gpu_num[a]} '--time='${t}
                                            }
                                   done
                                elif [ $p == 2 ]
                                then
                                   echo "Run Curv with Csd"
                                   for((c=0; c<csd_length; c++))
                                        do
                                            {
                                                python ${paras[p]} ${hp} '--n_epoch='${n_epoch} '--alpha='${alpha[a]} '--csd_importance='${csd_curv[c]} '--gpu='${gpu_num[a]} '--time='${t}
                                            }
                                   done

                                else
                                   echo "Run Prox with Csd"
                                   for((c=0; c<csd_length; c++))
                                        do
                                            {
                                                python ${paras[p]} ${hp} '--n_epoch='${n_epoch} '--alpha='${alpha[a]} '--csd_importance='${csd_prox[c]} '--gpu='${gpu_num[a]} '--time='${t}
                                            }
                                   done
                                fi

                            }
                    done
                }
        done
    }&
done
wait