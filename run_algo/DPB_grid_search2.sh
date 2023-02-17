for i in $(seq 1 9); do


    for g in $(seq 1 4); do

        #for d in $(seq 1 5); do


        for s in $(seq 1 6); do

            for q in $(seq 1 10); do

        

            z=$( echo "scale=5; 0.001*$i" | bc -l)
            x=$( echo "scale=5; 0.925+0.005*$g" | bc -l)
            #c=$( echo "scale=5; $d/10" | bc -l)
            v=$( echo "scale=5; 0.1*$s" | bc -l)


            python3 run_experiment.py -t avoid -w $z -g $x -d 0.7 -l $v -s $q

            done

        done


        #done


    done

done
