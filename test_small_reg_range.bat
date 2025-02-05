for bound in 5 10
do
    for epoch in 500 1000
    do
        for layer in 16@8 32@16 64@32
        do
            struct=${bound}@${layer}
            for regterm in 1e-3 1e-4 1e-5
            do
                for range in 1.0 1.5 2.0 2.5
                do
                    python experiment.py -lr 1e-4 -s $struct -e $epoch -wd $regterm -r range
                done
            done
        done
    done
done    