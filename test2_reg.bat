for bound in 5 10
do
    for epoch in 500 1000
    do
        for layer in 128@64 256@128 512@256
        do
            struct=${bound}@${layer}
            for lr in 5e-4 2e-4 1e-4
            do
                for regterm in 1e-3 1e-4 1e-5 1e-6 1e-7
                do
                    echo $lr $struct $epoch $regterm
                    python experiment.py -lr $lr -s $struct -e $epoch -wd $regterm
                done
            done 
        done
    done
done