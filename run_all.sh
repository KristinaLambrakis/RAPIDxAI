version=$1
if [ $# -eq 0 ]
then
  version=v5
else
  version=$1
fi

source build_image.sh "${version}"
source run_container.sh "${version}"

sleep 5s
source run_tester.sh "${version}"