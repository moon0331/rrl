for bound in 5 10
do
    for epoch in 500 1000
    do
        for layer in 16 32 64 16@8 32@16 64@32
        do
            struct=${bound}@${layer}
            for lr in 5e-4 2e-4 1e-4
            do
                for regterm in 1e-3 1e-4 1e-5 1e-6 0
                do
                    for range in 1.0 1.5 2.5 3.0
                    do
                        echo python experiment.py -lr $lr -s $struct -e $epoch -r $range -wd $regterm
                        python experiment.py -lr $lr -s $struct -e $epoch -r $range -wd $regterm -i 0
                        # variable range
                    done
                done
            done 
        done
    done
done
echo test_small.bat done