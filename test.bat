for bound in 5 10
do
    for epoch in 500 1000
    do
        for layer in 128 256 512
        do
            struct=${bound}@${layer}
            for lr in 5e-4 2e-4 1e-4
            do
                python experiment.py -lr $lr -s $struct -e $epoch
            done 
        done
    done
done