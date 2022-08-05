for bound in 5 10
do
    for epoch in 500 1000
    do
        for layer in 128 256 512 128@64 256@128 512@256
        do
            struct=${bound}@${layer}
            for lr in 5e-4 2e-4 1e-4
            do
                for regterm in 1e-3 1e-4 1e-5 1e-6 0
                do
                    echo python experiment.py -lr $lr -s $struct -e $epoch -r 2.0 -wd $regterm -i 1
                    python experiment.py -lr $lr -s $struct -e $epoch -r 2.0 -wd $regterm -i 1
                    # fix range -2 to 2
                done
            done 
        done
    done
done
echo test_large.bat done