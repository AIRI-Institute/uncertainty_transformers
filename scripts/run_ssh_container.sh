data_path=`realpath $1`
docker run --rm -d --name ue_results -v $data_path:/ue_data -p 5566:22 rastasheep/ubuntu-sshd

