BERTS="ALBERT BERT-LARGE MEDBERT MODERNBERT"
project_name="BERT"
entity_name="newchat"
metrics="writing-style completeness relevance"
folds=(0 1 2 3 4 5)

for bert in $BERTS;do
    #per model loop
    for m in $metrics;do
        #per metric loop
        for f in "${folds[@]}";do
            echo model:$bert, metric:$m,fold:$f
            
        done
    done
done