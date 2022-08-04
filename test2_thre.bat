for bound in 5 10
do
    for epoch in 500 1000
    do
        for layer in 128@64 256@128 512@256
        do
            struct=${bound}@${layer}
            for lr in 5e-4 2e-4 1e-4
            do
                for thr in 0.5 0.55 0.6 0.65 0.7
                do
                    echo $lr $struct $epoch $thr
                    python experiment.py -lr $lr -s $struct -e $epoch -t $thr
                done
            done 
        done
    done
done