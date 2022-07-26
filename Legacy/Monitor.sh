#!bin/bash 

rm -rf _Pickle 
rm -rf _TestResults
rm -rf _Cache
rm -rf _Models
rm -rf PresentationPlots

mkdir logs
mkdir _TestResults

source ~/anaconda3/etc/profile.d/conda.sh
conda activate GNN
python main.py >> logs/out.log & 

PID="$!"
cd _TestResults
while true; 
do
  cat ./*/*.txt > Out.txt
  cat Out.txt

  working=false
  if [[ "$(cat Out.txt)" != *"(-)"* ]]
  then 
    working=true
  fi 
  
  rm Out.txt
  if ps -p $PID > /dev/null
  then 
    echo "Running"
  else
    break
  fi
  sleep 2
  clear
done
cd ../

rm -rf _Pickle 
rm -rf _TestResults
rm -rf _Cache
rm -rf _Models
rm -rf PresentationPlots

if [[ $working == true ]]
then 
  echo "Going to COMMIT TO GIT AND SHUTTING DOWN SOON!"
  sleep 60
  rm -rf logs 
  git add . 
  git commit -m "Passed All Tests."
  git push origin RefactoringCode
  git push origin_github RefactoringCode
else
  echo "Going to COMMIT TO GIT AND SHUTTING DOWN SOON!"
  sleep 60
  rm -rf logs 
  git add . 
  git commit -m "Failed Some Tests."
  git push origin RefactoringCode
  git push origin_github RefactoringCode
  echo "Shutting down with commit."
fi
