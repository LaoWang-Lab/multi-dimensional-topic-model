for M in `seq 1 10`
do
  echo "$M"
  python mlda.py
  find trainedTopic -type d -iregex "trainedTopic/.+" | grep -v "run" | xargs -I '{}' mv '{}' "{}_run${M}"
done  
