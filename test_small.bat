for bound in 3 5 10
do
    for epoch in 500 1000
    do
        for layer in 16 32 64
        do
            struct=${bound}@${layer}
            for lr in 5e-4 2e-4 1e-4
            do
                python experiment.py -lr $lr -s $struct -e $epoch
            done 
        done
    done
done

for bound in 5 10
do
    for epoch in 500 1000
    do
        for layer in 16@8 32@16 64@32
        do
            struct=${bound}@${layer}
            for lr in 5e-4 2e-4 1e-4
            do
                python experiment.py -lr $lr -s $struct -e $epoch
            done 
        done
    done
done    